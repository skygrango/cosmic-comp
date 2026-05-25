// SPDX-License-Identifier: GPL-3.0-only

use crate::backend::kms::surface::ThreadCommand;
use smithay::backend::drm::{DrmDeviceNotifier, DrmEvent};
use smithay::reexports::calloop::{EventLoop, channel, channel::Sender};
use smithay::reexports::drm::control::crtc;
use std::collections::HashMap;

#[derive(Debug)]
pub enum KmsMessage {
    RegisterSurface(crtc::Handle, Sender<ThreadCommand>),
    UnregisterSurface(crtc::Handle),
}

pub fn start_kms_thread(notifier: DrmDeviceNotifier) -> Sender<KmsMessage> {
    let (tx, rx) = channel::channel();
    std::thread::Builder::new()
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
                    tracing::warn!("KMS Thread: Failed to gain real time thread priority (Check CAP_SYS_NICE)");
                }
            }

            let mut event_loop = EventLoop::try_new().unwrap();
            let handle = event_loop.handle();
            let mut surfaces = HashMap::new();

            handle.insert_source(notifier, move |event, metadata, surfaces: &mut HashMap<crtc::Handle, Sender<ThreadCommand>>| match event {
                DrmEvent::VBlank(crtc) => {
                    if let Some(tx) = surfaces.get(&crtc) {
                        let _ = tx.send(ThreadCommand::VBlank(metadata.take()));
                    }
                }
                DrmEvent::Error(err) => {
                    tracing::warn!(?err, "KMS Thread: DRM error");
                }
            }).unwrap();

            handle.insert_source(rx, move |event, _, surfaces: &mut HashMap<crtc::Handle, Sender<ThreadCommand>>| match event {
                channel::Event::Msg(KmsMessage::RegisterSurface(crtc, tx)) => {
                    surfaces.insert(crtc, tx);
                }
                channel::Event::Msg(KmsMessage::UnregisterSurface(crtc)) => {
                    surfaces.remove(&crtc);
                }
                channel::Event::Closed => {}
            }).unwrap();

            event_loop.run(None, &mut surfaces, |_| {}).unwrap();
        }).unwrap();
    tx
}
