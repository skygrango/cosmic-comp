// SPDX-License-Identifier: GPL-3.0-only

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use arc_swap::ArcSwap;
use smithay::backend::drm::{DrmDeviceFd, DrmEventMetadata, DrmEventTime};
use smithay::reexports::calloop::channel::Sender;
use smithay::reexports::drm::control::{Device as ControlDevice, Event, crtc};
use smithay::reexports::drm::{Device as BasicDevice, DriverCapability};

use crate::backend::kms::surface::ThreadCommand;

pub type RouteVec = Vec<(crtc::Handle, Sender<ThreadCommand>)>;
pub type SharedRoutes = Arc<ArcSwap<RouteVec>>;

#[derive(Clone, Debug)]
pub struct KmsThreadHandle {
    routes: SharedRoutes,
}

impl KmsThreadHandle {
    pub fn new(routes: SharedRoutes) -> Self {
        Self { routes }
    }

    pub fn register(&self, crtc: crtc::Handle, tx: Sender<ThreadCommand>) {
        self.routes.rcu(|routes| {
            let mut new = (**routes).clone();
            new.retain(|(c, _)| *c != crtc);
            new.push((crtc, tx.clone()));
            Arc::new(new)
        });
    }

    pub fn unregister(&self, crtc: crtc::Handle) {
        self.routes.rcu(|routes| {
            let mut new = (**routes).clone();
            new.retain(|(c, _)| *c != crtc);
            Arc::new(new)
        });
    }
}

pub struct KmsThread {
    handle: KmsThreadHandle,
    stop_fd: Option<OwnedFd>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for KmsThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KmsThread")
            .field("handle", &self.handle)
            .finish()
    }
}

impl std::ops::Deref for KmsThread {
    type Target = KmsThreadHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for KmsThread {
    fn drop(&mut self) {
        if let Some(stop_fd) = self.stop_fd.take() {
            let val: u64 = 1;
            unsafe {
                libc::write(
                    stop_fd.as_raw_fd(),
                    &val as *const _ as *const libc::c_void,
                    std::mem::size_of::<u64>(),
                );
            }
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn start_kms_thread(fd: DrmDeviceFd) -> Result<KmsThread, std::io::Error> {
    let exit_fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if exit_fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stop_fd = unsafe { OwnedFd::from_raw_fd(exit_fd) };
    let thread_stop_fd = stop_fd.try_clone()?;

    let routes = Arc::new(ArcSwap::from_pointee(Vec::new()));
    let handle = KmsThreadHandle::new(routes.clone());

    let thread = std::thread::Builder::new()
        .name("kms-thread".into())
        .spawn(move || {
            unsafe {
                let min_priority = libc::sched_get_priority_min(libc::SCHED_FIFO);
                let sp = libc::sched_param {
                    sched_priority: min_priority,
                };
                if libc::pthread_setschedparam(
                    libc::pthread_self(),
                    libc::SCHED_FIFO | libc::SCHED_RESET_ON_FORK,
                    &sp,
                ) != 0
                {
                    tracing::warn!(
                        "KMS Thread: Failed to gain real time thread priority (Check CAP_SYS_NICE)"
                    );
                }
            }

            let has_monotonic = fd
                .get_driver_capability(DriverCapability::MonotonicTimestamp)
                .unwrap_or(0)
                == 1;

            let drm_raw_fd = fd.as_raw_fd();
            let stop_raw_fd = thread_stop_fd.as_raw_fd();

            let mut poll_fds = [
                libc::pollfd {
                    fd: drm_raw_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: stop_raw_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];

            loop {
                poll_fds[0].revents = 0;
                poll_fds[1].revents = 0;

                let ret = unsafe { libc::poll(poll_fds.as_mut_ptr(), 2, -1) };

                if ret < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    tracing::error!(?err, "KMS Thread: poll failed");
                    break;
                }

                if poll_fds[1].revents & (libc::POLLIN | libc::POLLERR | libc::POLLHUP) != 0 {
                    tracing::debug!("KMS Thread: Stop signal received, exiting");
                    break;
                }

                if poll_fds[0].revents & libc::POLLIN != 0 {
                    match fd.receive_events() {
                        Ok(events) => {
                            let current_routes = routes.load();
                            for event in events {
                                let (crtc, duration, frame) = match event {
                                    Event::PageFlip(e) => (e.crtc, e.duration, e.frame),
                                    Event::Vblank(e) => (e.crtc, e.time, e.frame),
                                    Event::Unknown(_) => continue,
                                };
                                let time = if has_monotonic {
                                    DrmEventTime::Monotonic(duration)
                                } else {
                                    DrmEventTime::Realtime(
                                        std::time::SystemTime::UNIX_EPOCH + duration,
                                    )
                                };
                                let metadata = DrmEventMetadata {
                                    time,
                                    sequence: frame,
                                };
                                if let Some((_, tx)) =
                                    current_routes.iter().find(|(c, _)| *c == crtc)
                                {
                                    let _ = tx.send(ThreadCommand::VBlank(Some(metadata)));
                                }
                            }
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(err) => {
                            tracing::warn!(?err, "KMS Thread: Error receiving drm events");
                        }
                    }
                }

                if poll_fds[0].revents & (libc::POLLERR | libc::POLLHUP) != 0 {
                    tracing::warn!("KMS Thread: DRM device hung up or error");
                    break;
                }
            }
        })?;

    Ok(KmsThread {
        handle,
        stop_fd: Some(stop_fd),
        thread: Some(thread),
    })
}
