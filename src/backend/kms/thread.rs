// SPDX-License-Identifier: GPL-3.0-only

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use parking_lot::Mutex;

use anyhow::Result;
use smithay::backend::drm::{DrmDeviceFd, DrmDeviceNotifier, DrmEvent, compositor::DrmSubmission};
use smithay::reexports::calloop::{EventLoop, channel::{Sender, Channel, Event}, Interest, Mode, PostAction, generic::Generic};
use smithay::reexports::drm::control::crtc;
use tracing::{error, warn};

use super::surface::{Feedback, GbmDrmOutput, ThreadCommand};

/// Frame data for a KMS commit
pub struct KmsFrame {
    pub crtc: crtc::Handle,
    pub submission: Option<DrmSubmission>,
    pub fence: Option<OwnedFd>,
    pub feedback: Option<Feedback>,
}

/// Commands to the KMS thread
pub enum KmsMessage {
    /// Commit a frame to a CRTC
    Commit(KmsFrame),
    /// Register a compositor for a CRTC
    RegisterCompositor(crtc::Handle, Arc<Mutex<GbmDrmOutput>>),
    /// Register a sender for a CRTC's vblank events
    RegisterVBlankSender(crtc::Handle, Sender<ThreadCommand>),
    /// Shutdown the thread
    Shutdown,
}

impl std::fmt::Debug for KmsMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KmsMessage::Commit(frame) => f.debug_struct("Commit").field("crtc", &frame.crtc).finish(),
            KmsMessage::RegisterCompositor(crtc, _) => f.debug_struct("RegisterCompositor").field("crtc", crtc).finish(),
            KmsMessage::RegisterVBlankSender(crtc, _) => f.debug_struct("RegisterVBlankSender").field("crtc", crtc).finish(),
            KmsMessage::Shutdown => write!(f, "Shutdown"),
        }
    }
}

#[derive(Debug)]
pub struct KmsThread {
    sender: Sender<KmsMessage>,
}

impl KmsThread {
    pub fn spawn(
        fd: DrmDeviceFd,
        notifier: DrmDeviceNotifier,
    ) -> Result<Self> {
        let (tx, rx) = smithay::reexports::calloop::channel::channel();
        
        std::thread::Builder::new()
            .name("kms-manager".into())
            .spawn(move || {
                if let Err(err) = kms_thread_main(fd, notifier, rx) {
                    error!("KMS thread crashed: {}", err);
                }
            })?;

        Ok(KmsThread {
            sender: tx,
        })
    }

    pub fn sender(&self) -> Sender<KmsMessage> {
        self.sender.clone()
    }
}

struct KmsThreadState {
    vblank_senders: HashMap<crtc::Handle, Sender<ThreadCommand>>,
    compositors: HashMap<crtc::Handle, Arc<Mutex<GbmDrmOutput>>>,
}

fn kms_thread_main(
    _fd: DrmDeviceFd,
    notifier: DrmDeviceNotifier,
    receiver: Channel<KmsMessage>,
) -> Result<()> {
    // Set real-time priority
    unsafe {
        let min_priority = libc::sched_get_priority_min(libc::SCHED_RR);
        let sp = libc::sched_param {
            sched_priority: min_priority,
        };
        if libc::pthread_setschedparam(
            libc::pthread_self(),
            libc::SCHED_RR | libc::SCHED_RESET_ON_FORK,
            &sp,
        ) != 0
        {
            warn!("Failed to gain real time thread priority for KMS thread");
        }
    }

    let mut event_loop = EventLoop::try_new()?;
    let handle = event_loop.handle();

    let mut state = KmsThreadState {
        vblank_senders: HashMap::new(),
        compositors: HashMap::new(),
    };

    // Listen for KMS events
    handle.insert_source(notifier, move |event, metadata, state: &mut KmsThreadState| {
        match event {
            DrmEvent::VBlank(crtc) => {
                if let Some(sender) = state.vblank_senders.get(&crtc) {
                    let _ = sender.send(ThreadCommand::VBlank(metadata.take()));
                }
            }
            DrmEvent::Error(err) => {
                error!("DRM Error in KMS thread: {}", err);
            }
        }
    }).map_err(|err| anyhow::anyhow!("Failed to insert notifier: {}", err))?;

    // Listen for commands
    let loop_handle = handle.clone();
    handle.insert_source(receiver, move |event, _, state: &mut KmsThreadState| {
        match event {
            Event::Msg(KmsMessage::Commit(mut frame)) => {
                let crtc = frame.crtc;
                let mut submission = frame.submission.take();
                
                if let Some(fence) = frame.fence {
                    let res = loop_handle.insert_source(Generic::new(fence, Interest::READ, Mode::Level), move |_, _, state| {
                        let result = submission.take().unwrap().execute();
                        if let Some(sender) = state.vblank_senders.get(&crtc) {
                            let _ = sender.send(ThreadCommand::CommitDone(result));
                        }
                        Ok(PostAction::Remove)
                    });
                    if let Err(err) = res {
                        error!("Failed to insert fence source into KMS event loop: {}", err);
                    }
                } else {
                    let result = submission.take().unwrap().execute();
                    if let Some(sender) = state.vblank_senders.get(&crtc) {
                        let _ = sender.send(ThreadCommand::CommitDone(result));
                    }
                }
            }
            Event::Msg(KmsMessage::RegisterCompositor(crtc, compositor)) => {
                state.compositors.insert(crtc, compositor);
            }
            Event::Msg(KmsMessage::RegisterVBlankSender(crtc, sender)) => {
                state.vblank_senders.insert(crtc, sender);
            }
            Event::Msg(KmsMessage::Shutdown) => {
                // TODO: handle shutdown
            }
            Event::Closed => {
                // Receiver closed, shutdown
            }
        }
    }).map_err(|err| anyhow::anyhow!("Failed to insert receiver: {}", err))?;

    event_loop.run(None, &mut state, |_| {}).map_err(Into::into)
}
