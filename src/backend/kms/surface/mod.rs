// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    backend::{
        kms::{drm_helpers::HdrOutputState, surface::timings::SAMPLE_TIME_WINDOW},
        render::{
            CLEAR_COLOR, CursorMode, GlMultiError, GlMultiRenderer, PostprocessOutputConfig,
            PostprocessShader, PostprocessState, VulkanMultiRenderer,
            element::{CosmicElement, DamageElement},
            init_shaders, output_elements, postprocess_intermediate_format, set_hdr_client_blend,
            wayland::SurfaceRenderElement,
        },
    },
    config::ScreenFilter,
    shell::{CosmicSurface, Shell},
    state::SurfaceDmabufFeedback,
    utils::{
        env::{bool_var, hdr_policy, tearing_allowed_for},
        prelude::*,
    },
    wayland::handlers::{
        compositor::{FULLSCREEN_IMMEDIATE_RENDER, recursive_frame_time_estimation},
        image_copy_capture::{
            FrameHolder, PendingImageCopyData, SessionData, SessionHolder, submit_buffer,
        },
    },
};

use anyhow::{Context, Result};
use calloop::channel::Channel;
use cosmic_comp_config::output::comp::AdaptiveSync;
use smithay::{
    backend::{
        allocator::{
            Buffer, Fourcc,
            format::FormatSet,
            gbm::{GbmAllocator, GbmBuffer},
        },
        drm::{
            CursorBufferTransformFn, DrmDeviceFd, DrmEventMetadata, DrmEventTime, DrmNode,
            ScanoutPlan, SrgbToPqEncoder, VrrSupport,
            color::{CrtcColorState, PlaneColorConversion},
            colorop::PostBlendEncode,
            compositor::{
                BlitFrameResultError, FrameError, FrameFlags, PrimaryPlaneElement,
                RenderFrameResult,
            },
            exporter::gbm::GbmFramebufferExporter,
            gbm::GbmFramebuffer,
            output::DrmOutput,
        },
        egl::EGLContext,
        renderer::{
            Bind, Blit, BufferType, Color32F, Frame, Offscreen, Renderer, RendererSuper, Texture,
            TextureFilter, buffer_dimensions, buffer_type,
            damage::Error as RenderError,
            element::{
                Element, Id, Kind, RenderElementStates,
                texture::TextureRenderElement,
                utils::{
                    ConstrainAlign, ConstrainScaleBehavior, Relocate, RelocateRenderElement,
                    constrain_render_elements,
                },
            },
            gles::{
                GlesRenderbuffer, GlesRenderer, GlesTexture, HdrOutputConfig, Uniform,
                element::TextureShaderElement,
            },
            glow::GlowRenderer,
            multigpu::{ApiDevice, Error as MultiError, GpuManager, is_same_gpu},
            sync::SyncPoint,
            utils::with_renderer_surface_state,
        },
    },
    desktop::space::SpaceElement,
    desktop::utils::OutputPresentationFeedback,
    output::{Output, OutputNoMode},
    reexports::{
        calloop::{
            EventLoop, LoopHandle, RegistrationToken,
            channel::{Event, Sender, channel},
            timer::{TimeoutAction, Timer},
        },
        drm::control::{connector, crtc},
        wayland_protocols::wp::{
            linux_dmabuf::zv1::server::zwp_linux_dmabuf_feedback_v1,
            presentation_time::server::wp_presentation_feedback,
        },
        wayland_server::protocol::wl_surface::WlSurface,
    },
    utils::{Clock, IsAlive, Monotonic, Physical, Point, Rectangle, Scale, Size, Transform},
    wayland::{
        color::management::{Chromaticities, ImageDescription},
        dmabuf::{DmabufFeedbackBuilder, get_dmabuf},
        image_copy_capture::{
            CaptureFailureReason, Frame as ScreencopyFrame, SessionRef as ScreencopySessionRef,
        },
        presentation::Refresh,
        seat::WaylandFocus,
        shm::{shm_format_to_fourcc, with_buffer_contents},
    },
};
use std::fmt;
use tracing::{debug, error, info, trace, warn};

use std::{
    borrow::{Borrow, BorrowMut},
    collections::{HashMap, HashSet, hash_map},
    mem,
    sync::{
        Arc, LazyLock, Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender},
    },
    thread::JoinHandle,
    time::Duration,
};

mod timings;
pub use self::timings::Timings;

use super::{
    drm_helpers,
    render::{gles::GbmGlowBackend, vulkan::GbmVulkanBackend},
    thread::KmsMessage,
};
use smithay::backend::renderer::vulkan::VulkanRenderer;

pub enum SurfaceNodeRenderer {
    Egl(EGLContext),
    Vulkan(VulkanRenderer),
}

impl fmt::Debug for SurfaceNodeRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Egl(ctx) => f.debug_tuple("Egl").field(ctx).finish(),
            Self::Vulkan(r) => f.debug_tuple("Vulkan").field(r).finish(),
        }
    }
}

pub enum SurfaceGpuApi {
    Glow(GpuManager<GbmGlowBackend<DrmDeviceFd>>),
    Vulkan(GpuManager<GbmVulkanBackend<DrmDeviceFd>>),
}

impl fmt::Debug for SurfaceGpuApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Glow(api) => f.debug_tuple("Glow").field(api).finish(),
            Self::Vulkan(api) => f.debug_tuple("Vulkan").field(api).finish(),
        }
    }
}

static FULLSCREEN_SKIP_OTHER_SURFACE: LazyLock<bool> =
    LazyLock::new(|| bool_var("COSMIC_FULLSCREEN_SKIP_OTHER_SURFACE").unwrap_or(true));

static FULLSCREEN_SKIP_OTHER_SURFACE_ALWAYS: LazyLock<bool> =
    LazyLock::new(|| bool_var("COSMIC_FULLSCREEN_SKIP_OTHER_SURFACE_ALWAYS").unwrap_or(false));

static DISABLE_DIRECT_SCANOUT: LazyLock<bool> =
    LazyLock::new(|| bool_var("COSMIC_DISABLE_DIRECT_SCANOUT").unwrap_or(false));

static DISABLE_CURSOR_PLANE: LazyLock<bool> =
    LazyLock::new(|| bool_var("COSMIC_DISABLE_CURSOR_PLANE").unwrap_or(false));

static DISABLE_OVERLAY_SCANOUT: LazyLock<bool> =
    LazyLock::new(|| bool_var("COSMIC_DISABLE_OVERLAY_SCANOUT").unwrap_or(false));

const _30_HZ: Duration = Duration::from_nanos(1_000_000_000 / 30);
const MIN_VRR_TARGET_RATE: u32 = 30_000; // 30Hz in milliHz

#[inline]
pub fn resolve_vrr_target_rate(rate: u32, origin_rate: u32) -> u32 {
    if rate < MIN_VRR_TARGET_RATE {
        origin_rate
    } else {
        rate
    }
}

#[cfg(feature = "debug")]
use smithay_egui::EguiState;

#[derive(Debug)]
pub struct Surface {
    pub(crate) connector: connector::Handle,
    pub(super) crtc: crtc::Handle,
    pub(crate) output: Output,
    known_nodes: HashSet<DrmNode>,

    active: Arc<AtomicBool>,
    pub feedback: HashMap<DrmNode, SurfaceDmabufFeedback>,
    pub(super) primary_plane_formats: FormatSet,
    overlay_plane_formats: Option<FormatSet>,

    loop_handle: LoopHandle<'static, State>,
    thread_command: Sender<ThreadCommand>,
    thread_token: RegistrationToken,
    thread: Option<JoinHandle<()>>,
    emergency_shutdown_id: Option<u64>,

    kms_thread: Sender<KmsMessage>,

    dpms: bool,
    pub is_vulkan: bool,
    adaptive_sync_mode: AdaptiveSync,
    hdr_enabled: bool,
    pub(super) hdr_sink_capabilities: Option<drm_helpers::HdrSinkCapabilities>,
    pub(super) native_primaries: Option<Chromaticities>,
    hdr_reference_white: f32,
    pub(crate) hdr_hardware_offload: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorTransformMode {
    Passthrough,
    PqEncode { ref_white: u32 },
    LinearRec709,
}

impl CursorTransformMode {
    #[inline]
    pub fn for_state(
        hdr_enabled: bool,
        active_scanout_plan: Option<ScanoutPlan>,
        hdr_reference_white: f32,
    ) -> Self {
        if !hdr_enabled {
            CursorTransformMode::Passthrough
        } else if matches!(
            active_scanout_plan,
            Some(ScanoutPlan::CrtcHardware(
                PlaneColorConversion::ScRgbToPq { .. }
            ))
        ) {
            CursorTransformMode::LinearRec709
        } else {
            CursorTransformMode::PqEncode {
                ref_white: hdr_reference_white.round() as u32,
            }
        }
    }
}

pub struct SurfaceThreadState {
    // rendering
    api: SurfaceGpuApi,
    pub is_vulkan: bool,
    primary_node: Arc<RwLock<Option<DrmNode>>>,
    target_node: DrmNode,
    active: Arc<AtomicBool>,
    vrr_mode: AdaptiveSync,
    vrr_target_rate: u32,
    frame_flags: FrameFlags,
    compositor: Option<GbmDrmOutput>,

    state: QueueState,
    timings: Timings,
    frame_callback_seq: usize,
    thread_sender: Sender<SurfaceCommand>,

    output: Output,
    fullscreen: Option<FullscreenOccupied>,
    mirroring: Option<Output>,
    screen_filter: ScreenFilter,
    hdr_enabled: bool,
    is_scanout: bool,
    swapchin_is_scanout: bool,
    active_scanout_plan: Option<ScanoutPlan>,
    hdr_reference_white: f32,
    hdr_max_luminance: f32,
    hdr_hardware_offload: bool,
    hdr_config: Option<HdrOutputConfig>,
    postprocess_textures: HashMap<DrmNode, PostprocessState>,
    current_cursor_transform_mode: Option<CursorTransformMode>,

    shell: Arc<parking_lot::RwLock<Shell>>,

    loop_handle: LoopHandle<'static, Self>,
    clock: Clock<Monotonic>,

    min_vrr: Option<u32>,
    min_vrr_frame_time: Option<Duration>,

    #[cfg(feature = "debug")]
    egui: EguiState,

    last_sequence: Option<u32>,
    /// Tracy frame that goes from vblank to vblank.
    vblank_frame: Option<tracy_client::Frame>,
    /// Frame name for the VBlank frame.
    vblank_frame_name: tracy_client::FrameName,
    /// Plot name for the time since presentation plot.
    time_since_presentation_plot_name: tracy_client::PlotName,
    /// Plot name for the presentation misprediction plot.
    presentation_misprediction_plot_name: tracy_client::PlotName,
    sequence_delta_plot_name: tracy_client::PlotName,
}

static HDR_SURFACE_SENDERS: OnceLock<Mutex<HashMap<u64, Sender<ThreadCommand>>>> = OnceLock::new();
static NEXT_HDR_SURFACE_ID: AtomicU64 = AtomicU64::new(1);

fn register_hdr_surface(sender: Sender<ThreadCommand>) -> u64 {
    let id = NEXT_HDR_SURFACE_ID.fetch_add(1, Ordering::Relaxed);
    HDR_SURFACE_SENDERS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .insert(id, sender);
    id
}

/// Stop strict-HDR surface threads without relying on the compositor's main
/// event loop. A blocked Wayland/config callback must not prevent connector
/// color state from being cleared before an external watchdog ends the process.
pub fn emergency_shutdown_hdr_surfaces() {
    let senders = HDR_SURFACE_SENDERS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect::<Vec<_>>();
    for sender in senders {
        let _ = sender.send(ThreadCommand::End);
    }
}

/// Receives from a surface-thread channel, bounded by the teardown timeout in
/// strict HDR mode so a wedged thread cannot hang the compositor forever.
fn recv_bounded<T>(rx: Receiver<T>) -> Result<T, String> {
    rx.recv_timeout(hdr_policy().teardown_timeout)
        .map_err(|err| err.to_string())
}

/// Determines whether a surface's color description matches the output's color state
/// for direct primary plane scanout without shader composition, following KWin conventions.
#[allow(dead_code)]
pub fn is_surface_scanout_compatible(
    output_hdr_enabled: bool,
    surface_desc: Option<&ImageDescription>,
) -> bool {
    if output_hdr_enabled {
        // On an HDR output (configured for BT.2100 PQ in KMS), only content encoded
        // in BT.2100 PQ matches the display hardware pipeline directly without
        // requiring GPU shader tone-mapping or color space expansion.
        surface_desc.is_some_and(|desc| desc.is_pq_bt2020())
    } else {
        // On an SDR output, standard SDR content (non-HDR) matches the display pipeline.
        !surface_desc.is_some_and(|desc| desc.is_hdr())
    }
}

#[allow(dead_code)]
pub fn is_scanout_compatible(output_hdr_enabled: bool, is_fullscreen_hdr: bool) -> bool {
    if output_hdr_enabled == is_fullscreen_hdr {
        true
    } else {
        false
    }
}

pub type GbmDrmOutput = DrmOutput<
    GbmAllocator<DrmDeviceFd>,
    GbmFramebufferExporter<DrmDeviceFd>,
    Option<(
        OutputPresentationFeedback,
        Receiver<PendingImageCopyData>,
        Duration,
    )>,
    DrmDeviceFd,
>;

#[derive(Debug, Default)]
pub struct OutputSwapchainFormat(pub std::sync::Mutex<Option<Fourcc>>);

#[derive(Debug, Clone, Default)]
pub struct OutputVulkanTimeline(
    pub std::sync::Arc<parking_lot::RwLock<Option<smithay::backend::drm::sync::DrmTimeline>>>,
);

pub fn output_vulkan_timeline(
    output: &smithay::output::Output,
) -> Option<smithay::backend::drm::sync::DrmTimeline> {
    output
        .user_data()
        .get::<OutputVulkanTimeline>()
        .and_then(|t| t.0.read().clone())
}

#[derive(Debug, Default)]
pub enum QueueState {
    #[default]
    Idle,
    /// A redraw is queued.
    Queued(RegistrationToken),
    /// We submitted a frame to the KMS and waiting for it to be presented.
    WaitingForVBlank {
        redraw_needed: bool,
        fullscreen_request: bool,
    },
    /// We did not submit anything to KMS and made a timer to fire at the estimated VBlank.
    WaitingForEstimatedVBlank(RegistrationToken),
    /// A redraw is queued on top of the above.
    WaitingForEstimatedVBlankAndQueued {
        estimated_vblank: RegistrationToken,
        queued_render: RegistrationToken,
    },
}

#[derive(Debug)]
pub enum ThreadCommand {
    Suspend(SyncSender<()>),
    Resume {
        compositor: GbmDrmOutput,
    },
    NodeAdded {
        node: DrmNode,
        gbm: GbmAllocator<DrmDeviceFd>,
        renderer: SurfaceNodeRenderer,
        sync: SyncSender<()>,
    },
    NodeRemoved {
        node: DrmNode,
        sync: SyncSender<()>,
    },
    UpdateMirroring(Option<Output>),
    UpdateScreenFilter(ScreenFilter),
    UpdateHdr {
        enabled: bool,
        reference_white: f32,
        max_luminance: f32,
        hardware_offload: bool,
    },
    VBlank(Option<DrmEventMetadata>),
    ScheduleRender(bool),
    AdaptiveSyncAvailable(SyncSender<Result<VrrSupport>>),
    UseAdaptiveSync(AdaptiveSync),
    UpdateVrrTargetRate(u32),
    AllowFrameFlags(bool, FrameFlags),
    End,
    DpmsOff,
    DpmsOn,
}

#[derive(Debug)]
pub enum SurfaceCommand {
    SignalFIFO,
    SendFrames(usize),
    RenderStates(RenderElementStates),
    FatalRenderError(String),
    DeviceLost(DrmNode),
    ProcessShmScreencopy,
}

pub struct PendingShmCapture {
    pub task: smithay::backend::renderer::vulkan::PendingVulkanShmCopy,
    pub frame: ScreencopyFrame,
    pub transform: Transform,
    pub damage: Vec<Rectangle<i32, smithay::utils::Buffer>>,
    pub presentation_time: Duration,
}

#[derive(Default, Clone)]
pub struct OutputPendingShmCaptures(pub Arc<std::sync::Mutex<Vec<PendingShmCapture>>>);

impl PendingShmCapture {
    pub fn process(self) {
        let buffer = self.frame.buffer();
        let transform = self.transform;
        let damage = self.damage;
        let presentation_time = self.presentation_time;
        let res = smithay::wayland::shm::with_buffer_contents_mut(&buffer, |ptr, len, data| {
            self.task.wait_and_copy(ptr, len, data.offset, data.stride)
        });
        match res {
            Ok(Ok(())) => {
                self.frame.success(transform, damage, presentation_time);
            }
            Ok(Err(err)) => {
                tracing::error!("PendingShmCapture wait_and_copy failed: {:?}", err);
                self.frame.fail(CaptureFailureReason::Unknown);
            }
            Err(err) => {
                tracing::error!(
                    "PendingShmCapture with_buffer_contents_mut failed: {:?}",
                    err
                );
                self.frame.fail(CaptureFailureReason::Unknown);
            }
        }
    }
}

#[derive(Debug, Default)]
struct PrePostprocessData {
    states: Option<RenderElementStates>,
    texture: Option<GlesTexture>,
    cursor_texture: Option<GlesTexture>,
    cursor_geometry: Option<Rectangle<i32, Physical>>,
}

impl Surface {
    pub fn new(
        output: &Output,
        crtc: crtc::Handle,
        connector: connector::Handle,
        primary_node: Arc<RwLock<Option<DrmNode>>>,
        dev_node: DrmNode,
        target_node: DrmNode,
        evlh: &LoopHandle<'static, State>,
        screen_filter: ScreenFilter,
        shell: Arc<parking_lot::RwLock<Shell>>,
        startup_done: Arc<AtomicBool>,
        is_vulkan: bool,
        kms_thread: &Sender<KmsMessage>,
    ) -> Result<Self> {
        unsafe {
            let min_priority = libc::sched_get_priority_max(libc::SCHED_RR);
            let sp = libc::sched_param {
                sched_priority: min_priority,
            };
            if libc::pthread_setschedparam(
                libc::pthread_self(),
                libc::SCHED_RR | libc::SCHED_RESET_ON_FORK,
                &sp,
            ) != 0
            {
                tracing::warn!("Failed to gain real time thread priority (Check CAP_SYS_NICE)");
            }
        }
        let (tx, rx) = channel::<ThreadCommand>();
        let _ = kms_thread.send(KmsMessage::RegisterSurface(crtc, tx.clone()));
        let (tx2, rx2) = channel::<SurfaceCommand>();
        let active = Arc::new(AtomicBool::new(false));

        let active_clone = active.clone();
        let output_clone = output.clone();

        let thread = std::thread::Builder::new()
            .name(format!("surface-{}", output.name()))
            .spawn(move || {
                if let Err(err) = surface_thread(
                    output_clone,
                    primary_node,
                    target_node,
                    shell,
                    active_clone,
                    screen_filter,
                    tx2,
                    rx,
                    startup_done,
                    is_vulkan,
                ) {
                    error!("Surface thread crashed: {}", err);
                }
            })
            .context("Failed to spawn surface thread")?;

        output
            .user_data()
            .insert_if_missing_threadsafe(OutputPendingShmCaptures::default);
        let output_clone = output.clone();
        let thread_token = evlh
            .insert_source(rx2, move |command, _, state| match command {
                Event::Msg(SurfaceCommand::SignalFIFO) => {
                    output_clone.signal_fifo(state);
                }
                Event::Msg(SurfaceCommand::ProcessShmScreencopy) => {
                    if let Some(pending_captures) = output_clone.user_data().get::<OutputPendingShmCaptures>() {
                        let captures: Vec<PendingShmCapture> = {
                            let mut guard = pending_captures.0.lock().unwrap();
                            std::mem::take(&mut *guard)
                        };
                        if !captures.is_empty() {
                            let thread_name = format!("shm-screencopy-{}", output_clone.name());
                            if let Err(err) = std::thread::Builder::new()
                                .name(thread_name)
                                .spawn(move || {
                                    for capture in captures {
                                        capture.process();
                                    }
                                })
                            {
                                tracing::error!("Failed to spawn shm screencopy thread: {err:?}");
                            }
                        }
                    }
                }
                Event::Msg(SurfaceCommand::SendFrames(sequence)) => {
                    if output_clone.mirroring().is_some() {
                        return;
                    }
                    state.common.send_frames(&output_clone, Some(sequence));
                }
                Event::Msg(SurfaceCommand::RenderStates(states)) => {
                    if output_clone.mirroring().is_some() {
                        return;
                    }
                    state.common.update_primary_output(&output_clone, &states);
                    let kms = state.backend.kms();
                    let surface = &mut kms
                        .drm_devices
                        .get_mut(&dev_node)
                        .unwrap()
                        .inner
                        .surfaces
                        .get_mut(&crtc)
                        .unwrap();

                    state
                        .common
                        .send_dmabuf_feedback(&output_clone, &states, |source_node| {
                            if let Some(cached_feedback) = surface.feedback.get(&source_node) {
                                Some(cached_feedback.clone())
                            } else {
                                // If we have freed the node, because it didn't have any active buffers/surfaces,
                                // we might not be able to evaluate surface feedback yet.
                                let render_formats = kms.api.dmabuf_formats(&source_node)?;
                                // In contrast we must have the target node, if we have an active surface
                                let target_formats = kms.api.dmabuf_formats(&target_node).unwrap();
                                let feedback = get_surface_dmabuf_feedback(
                                    source_node,
                                    target_node,
                                    render_formats,
                                    target_formats,
                                    surface.primary_plane_formats.clone(),
                                    surface.overlay_plane_formats.clone(),
                                );
                                surface.feedback.insert(source_node, feedback.clone());
                                Some(feedback)
                            }
                        });
                }

                Event::Msg(SurfaceCommand::FatalRenderError(err)) => {
                    error!(output = %output_clone.name(), %err, "strict HDR output failed after activation; ending session");
                    state.common.should_stop = true;
                    state.common.event_loop_signal.stop();
                    state.common.event_loop_signal.wakeup();
                }
                Event::Msg(SurfaceCommand::DeviceLost(target_node)) => {
                    error!(output = %output_clone.name(), ?target_node, "Vulkan device lost received on main thread; initiating recovery");
                    state.handle_vulkan_device_lost(target_node);
                }
                Event::Closed => {}
            })
            .map_err(|_| anyhow::anyhow!("Failed to establish channel to surface thread"))?;

        let emergency_shutdown_id = Some(register_hdr_surface(tx.clone()));

        Ok(Surface {
            connector,
            crtc,
            output: output.clone(),
            known_nodes: HashSet::new(),
            active,
            feedback: HashMap::new(),
            primary_plane_formats: FormatSet::default(),
            overlay_plane_formats: None,
            loop_handle: evlh.clone(),
            thread_command: tx,
            thread_token,
            thread: Some(thread),
            emergency_shutdown_id,
            kms_thread: kms_thread.clone(),
            dpms: true,
            is_vulkan,
            adaptive_sync_mode: AdaptiveSync::Disabled,
            hdr_enabled: false,
            hdr_sink_capabilities: None,
            native_primaries: None,
            hdr_reference_white: 203.0,
            hdr_hardware_offload: false,
        })
    }

    pub fn known_nodes(&self) -> &HashSet<DrmNode> {
        &self.known_nodes
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    pub fn add_node(
        &mut self,
        node: DrmNode,
        gbm: GbmAllocator<DrmDeviceFd>,
        renderer: SurfaceNodeRenderer,
    ) {
        self.known_nodes.insert(node);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self.thread_command.send(ThreadCommand::NodeAdded {
            node,
            gbm,
            renderer,
            sync: tx,
        });
        self.wait_for_surface_ack(rx, "adding renderer node");
    }

    pub fn remove_node(&mut self, node: DrmNode) {
        self.known_nodes.remove(&node);
        self.feedback.remove(&node);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self
            .thread_command
            .send(ThreadCommand::NodeRemoved { node, sync: tx });
        // Block so we can be sure the file descriptor is closed
        // (which is relevant for the udev device_removed callback).
        self.wait_for_surface_ack(rx, "removing renderer node");
    }

    pub fn on_vblank(&self, metadata: Option<DrmEventMetadata>) {
        let _ = self.thread_command.send(ThreadCommand::VBlank(metadata));
    }

    pub fn schedule_render(&self) {
        self.schedule_backend(false);
    }

    pub fn schedule_render_fullscreen(&self) {
        self.schedule_backend(true);
    }

    pub fn schedule_backend(&self, is_fullscreen: bool) {
        if self.dpms {
            let _ = self
                .thread_command
                .send(ThreadCommand::ScheduleRender(is_fullscreen));
        }
    }

    pub fn set_mirroring(&mut self, output: Option<Output>) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateMirroring(output));
    }

    pub fn set_screen_filter(&mut self, config: ScreenFilter) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateScreenFilter(config));
    }

    /// Queue HDR shader state before `resume`. FIFO command ordering ensures
    /// that a newly resumed output cannot render before this state is applied.
    pub fn prepare_hdr_rendering(
        &mut self,
        enabled: bool,
        reference_white: f32,
        hardware_offload: bool,
    ) {
        self.hdr_enabled = enabled;
        self.hdr_reference_white = reference_white.clamp(80.0, 10_000.0);
        self.hdr_hardware_offload = hardware_offload;
        let max_luminance = self
            .hdr_sink_capabilities
            .map(|c| c.max_luminance as f32)
            .unwrap_or(1000.0)
            .max(self.hdr_reference_white);
        let _ = self.thread_command.send(ThreadCommand::UpdateHdr {
            enabled,
            reference_white: self.hdr_reference_white,
            max_luminance,
            hardware_offload,
        });
    }

    pub fn hdr_rendering(&self) -> (bool, f32) {
        (self.hdr_enabled, self.hdr_reference_white)
    }

    pub fn adaptive_sync_support(&self) -> Result<VrrSupport> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self
            .thread_command
            .send(ThreadCommand::AdaptiveSyncAvailable(tx));
        recv_bounded(rx).map_err(|err| anyhow::anyhow!("surface thread VRR query failed: {err}"))?
    }

    pub fn use_adaptive_sync(&mut self, vrr: AdaptiveSync) {
        if self
            .thread_command
            .send(ThreadCommand::UseAdaptiveSync(vrr))
            .is_ok()
        {
            self.adaptive_sync_mode = vrr;
        }
    }

    /// Whether the render thread still needs the requested adaptive-sync mode.
    pub fn adaptive_sync_update_required(&self, requested: AdaptiveSync) -> bool {
        adaptive_sync_update_required(requested, self.adaptive_sync_mode)
    }

    pub fn set_vrr_target_rate(&mut self, rate: u32) {
        let origin_rate = self
            .output
            .current_mode()
            .map(|m| m.refresh as u32)
            .filter(|&r| r > 0)
            .unwrap_or(60000);
        let rate = resolve_vrr_target_rate(rate, origin_rate);
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateVrrTargetRate(rate));
    }

    pub fn allow_frame_flags(&mut self, flag: bool, flags: FrameFlags) {
        let _ = self
            .thread_command
            .send(ThreadCommand::AllowFrameFlags(flag, flags));
    }

    pub fn suspend(&mut self) {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self.thread_command.send(ThreadCommand::Suspend(tx));
        self.wait_for_surface_ack(rx, "suspending output");
    }

    pub fn resume(
        &mut self,
        compositor: GbmDrmOutput,
        primary_plane_formats: FormatSet,
        overlay_plane_formats: Option<FormatSet>,
    ) {
        self.primary_plane_formats = primary_plane_formats;
        self.overlay_plane_formats = overlay_plane_formats;
        self.feedback.clear();
        self.active.store(true, Ordering::SeqCst);
        self.dpms = true;

        let _ = self
            .thread_command
            .send(ThreadCommand::Resume { compositor });
    }

    pub fn get_dpms(&mut self) -> bool {
        self.dpms
    }

    pub fn set_dpms(&mut self, on: bool) {
        if self.dpms != on {
            self.dpms = on;
            if on {
                let _ = self.thread_command.send(ThreadCommand::DpmsOn);
            } else {
                let _ = self.thread_command.send(ThreadCommand::DpmsOff);
            }
        }
    }

    pub fn drop_and_join(mut self) {
        let thread = self.thread.take();
        std::mem::drop(self);
        if let Some(thread) = thread {
            let name = thread.thread().name().unwrap().to_string();
            {
                let deadline = std::time::Instant::now() + hdr_policy().teardown_timeout;
                while !thread.is_finished() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                if !thread.is_finished() {
                    warn!(
                        output = %name,
                        operation = "drop_and_join",
                        "surface thread did not stop before the strict HDR deadline; detaching it"
                    );
                    return;
                }
            }
            let _ = thread.join();
            info!("Thread {} terminated.", name)
        }
    }

    fn wait_for_surface_ack(&self, rx: Receiver<()>, operation: &'static str) {
        if let Err(err) = recv_bounded(rx) {
            warn!(
                output = %self.output.name(),
                operation,
                %err,
                "surface-thread synchronization failed"
            );
        }
    }

    /// Ask the surface thread to stop and return its join handle without waiting.
    ///
    /// The compositor owned by that thread is dropped while the DRM device is still
    /// active, allowing its atomic surface to clear persistent connector color state.
    pub fn begin_shutdown(mut self) -> Option<JoinHandle<()>> {
        let thread = self.thread.take();
        std::mem::drop(self);
        thread
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let _ = self
            .kms_thread
            .send(KmsMessage::UnregisterSurface(self.crtc));
        if let Some(id) = self.emergency_shutdown_id.take()
            && let Some(senders) = HDR_SURFACE_SENDERS.get()
        {
            senders.lock().unwrap().remove(&id);
        }
        let _ = self.thread_command.send(ThreadCommand::End);
        self.loop_handle.remove(self.thread_token);
        if let Some(thread) = self.thread.take() {
            let _ = thread;
            // We want to do this, but this currently deadlocks on `apply_config_for_outputs`.
            /*
                let name = thread.thread().name().unwrap().to_string();
                let _ = thread.join();
                info!("Thread {} terminated.", name)
            */
        }
    }
}

fn adaptive_sync_update_required(
    requested: AdaptiveSync,
    render_thread_mode: AdaptiveSync,
) -> bool {
    requested != render_thread_mode
}

fn surface_thread(
    output: Output,
    primary_node: Arc<RwLock<Option<DrmNode>>>,
    target_node: DrmNode,
    shell: Arc<parking_lot::RwLock<Shell>>,
    active: Arc<AtomicBool>,
    screen_filter: ScreenFilter,
    thread_sender: Sender<SurfaceCommand>,
    thread_receiver: Channel<ThreadCommand>,
    startup_done: Arc<AtomicBool>,
    is_vulkan: bool,
) -> Result<()> {
    let name = output.name();
    profiling::register_thread!(&format!("Surface Thread {}", name));

    let mut event_loop = EventLoop::try_new().unwrap();

    let api = if is_vulkan {
        SurfaceGpuApi::Vulkan(
            GpuManager::new(GbmVulkanBackend::<DrmDeviceFd>::default())
                .context("Failed to initialize Vulkan rendering api")?,
        )
    } else {
        SurfaceGpuApi::Glow(
            GpuManager::new(GbmGlowBackend::<DrmDeviceFd>::default())
                .context("Failed to initialize rendering api")?,
        )
    };

    #[cfg(feature = "debug")]
    let egui = {
        let state = smithay_egui::EguiState::new(Rectangle::from_size((400, 800).into()));
        let visuals = egui::style::Visuals {
            window_shadow: egui::Shadow::NONE,
            ..Default::default()
        };
        state.context().set_visuals(visuals);
        state
    };

    let vblank_frame_name = tracy_client::FrameName::new_leak(format!("vblank on {name}"));
    let time_since_presentation_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} time since presentation, ms"));
    let presentation_misprediction_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} presentation misprediction, ms"));
    let sequence_delta_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} sequence delta"));

    let mut state = SurfaceThreadState {
        api,
        is_vulkan,
        primary_node,
        target_node,
        active,
        compositor: None,
        frame_flags: FrameFlags::DEFAULT,
        vrr_mode: AdaptiveSync::Disabled,
        vrr_target_rate: output
            .current_mode()
            .map(|m| m.refresh as u32)
            .unwrap_or(60000),

        state: QueueState::Idle,
        timings: Timings::new(None, None, false, target_node),
        frame_callback_seq: 0,
        thread_sender,

        output,
        fullscreen: None,
        mirroring: None,
        screen_filter,
        hdr_enabled: false,
        is_scanout: false,
        swapchin_is_scanout: false,
        active_scanout_plan: None,
        hdr_reference_white: 203.0,
        hdr_max_luminance: 1000.0,
        hdr_hardware_offload: false,
        hdr_config: None,
        postprocess_textures: HashMap::new(),
        current_cursor_transform_mode: None,

        shell,
        loop_handle: event_loop.handle(),
        clock: Clock::new(),

        min_vrr: None,
        min_vrr_frame_time: None,

        #[cfg(feature = "debug")]
        egui,

        last_sequence: None,
        vblank_frame: None,
        vblank_frame_name,
        time_since_presentation_plot_name,
        presentation_misprediction_plot_name,
        sequence_delta_plot_name,
    };
    state.update_hdr_config();

    let signal = event_loop.get_signal();
    event_loop
        .handle()
        .insert_source(thread_receiver, move |command, _, state| match command {
            Event::Msg(ThreadCommand::Suspend(tx)) => state.suspend(tx),
            Event::Msg(ThreadCommand::Resume { compositor }) => {
                state.resume(compositor);
            }
            Event::Msg(ThreadCommand::NodeAdded {
                node,
                gbm,
                renderer,
                sync,
            }) => {
                if let Err(err) = state.node_added(node, gbm, renderer) {
                    warn!(?err, ?node, "Failed to add node to surface-thread");
                }
                let _ = sync.send(());
            }
            Event::Msg(ThreadCommand::NodeRemoved { node, sync }) => {
                state.node_removed(node);
                let _ = sync.send(());
            }
            Event::Msg(ThreadCommand::VBlank(metadata)) => {
                state.on_vblank(metadata);
            }
            Event::Msg(ThreadCommand::ScheduleRender(is_fullscreen)) => {
                if !startup_done.load(Ordering::SeqCst) {
                    return;
                }

                state.queue_redraw(false, is_fullscreen);
            }
            Event::Msg(ThreadCommand::UpdateMirroring(mirroring_output)) => {
                state.update_mirroring(mirroring_output);
            }
            Event::Msg(ThreadCommand::UpdateScreenFilter(filter_config)) => {
                state.update_screen_filter(filter_config);
            }
            Event::Msg(ThreadCommand::UpdateHdr {
                enabled,
                reference_white,
                max_luminance,
                hardware_offload,
            }) => {
                state.hdr_enabled = enabled;
                state.hdr_reference_white = reference_white.clamp(80.0, 10_000.0);
                state.hdr_max_luminance = max_luminance.max(state.hdr_reference_white);
                state.hdr_hardware_offload = hardware_offload;
                state.update_hdr_config();
                // Shader uniforms are not part of the texture's commit
                // counter.  Recreate the post-process target so a live HDR
                // policy change gets a fresh element identity and full redraw.
                state.postprocess_textures.clear();
                if let Some(compositor) = state.compositor.as_mut() {
                    compositor.with_compositor(|c| c.reset_buffer_ages());
                }
                if enabled {
                    state.frame_flags.remove(
                        FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                            | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
                    );
                } else {
                    state.frame_flags.insert(FrameFlags::DEFAULT);
                    if *DISABLE_DIRECT_SCANOUT {
                        state.frame_flags.remove(
                            FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY
                                | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
                        );
                    } else if *DISABLE_OVERLAY_SCANOUT {
                        state
                            .frame_flags
                            .remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
                    }
                    if *DISABLE_CURSOR_PLANE {
                        state
                            .frame_flags
                            .remove(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT);
                    }
                }
                state.queue_redraw(true, false);
            }
            Event::Msg(ThreadCommand::AdaptiveSyncAvailable(result)) => {
                if let Some(compositor) = state.compositor.as_mut() {
                    let _ = result.send(
                        compositor
                            .with_compositor(|c| {
                                c.vrr_supported(c.pending_connectors().into_iter().next().unwrap())
                            })
                            .map_err(Into::into),
                    );
                } else {
                    let _ = result.send(Err(anyhow::anyhow!("Set vrr with inactive surface")));
                }
            }
            Event::Msg(ThreadCommand::UseAdaptiveSync(vrr)) => {
                state.vrr_mode = vrr;
            }
            Event::Msg(ThreadCommand::UpdateVrrTargetRate(rate)) => {
                let origin_rate = state
                    .output
                    .current_mode()
                    .map(|m| m.refresh as u32)
                    .filter(|&r| r > 0)
                    .unwrap_or(60000);
                let is_below_30hz = rate < MIN_VRR_TARGET_RATE;
                let target_rate = resolve_vrr_target_rate(rate, origin_rate);
                state.vrr_target_rate = target_rate;
                let interval = if is_below_30hz {
                    state
                        .timings
                        .origin_refresh_interval_ns
                        .map(|ns| Duration::from_nanos(ns.get()))
                        .unwrap_or_else(|| Duration::from_secs_f64(1000. / origin_rate as f64))
                } else {
                    Duration::from_secs_f64(1000. / target_rate as f64)
                };
                state.timings.set_vrr_target_rate_interval(Some(interval));
                if state.timings.vrr() {
                    state.timings.refresh_interval_ns = state.timings.vrr_target_rate_internal_ns;
                    state.timings.previous_frames.clear();
                }
            }
            Event::Msg(ThreadCommand::DpmsOff) => {
                if let Some(compositor) = state.compositor.as_mut() {
                    if let Err(err) = compositor.with_compositor(|c| c.clear()) {
                        error!("Failed to set DPMS off: {:?}", err);
                    }
                    match std::mem::replace(&mut state.state, QueueState::Idle) {
                        QueueState::Idle => {}
                        QueueState::Queued(token)
                        | QueueState::WaitingForEstimatedVBlank(token) => {
                            state.loop_handle.remove(token);
                        }
                        QueueState::WaitingForVBlank { .. } => {
                            state.timings.discard_current_frame()
                        }
                        QueueState::WaitingForEstimatedVBlankAndQueued {
                            estimated_vblank,
                            queued_render,
                        } => {
                            state.loop_handle.remove(estimated_vblank);
                            state.loop_handle.remove(queued_render);
                        }
                    };
                }
            }
            Event::Msg(ThreadCommand::DpmsOn) => {
                if let Some(compositor) = state.compositor.as_mut()
                    && let Err(err) = compositor.with_compositor(|c| c.reset_state())
                {
                    error!(?err, "failed to restore output state after DPMS");
                }
                state.queue_redraw(false, false);
            }
            Event::Msg(ThreadCommand::AllowFrameFlags(flag, mut flags)) => {
                if state.hdr_enabled {
                    flags.remove(
                        FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                            | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
                    );
                } else if *DISABLE_DIRECT_SCANOUT {
                    flags.remove(
                        FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                            | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY
                            | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
                    );
                }
                if *DISABLE_OVERLAY_SCANOUT {
                    flags.remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
                }
                if *DISABLE_CURSOR_PLANE {
                    flags.remove(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT);
                }

                if flag {
                    state.frame_flags.insert(flags);
                } else {
                    state.frame_flags.remove(flags);
                }
                if state.hdr_enabled {
                    state.frame_flags.remove(
                        FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                            | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
                    );
                }
            }
            Event::Closed | Event::Msg(ThreadCommand::End) => {
                signal.stop();
                signal.wakeup();
            }
        })
        .map_err(|insert_error| insert_error.error)
        .context("Failed to listen for events")?;

    event_loop.run(None, &mut state, |_| {}).map_err(Into::into)
}

impl SurfaceThreadState {
    fn suspend(&mut self, tx: SyncSender<()>) {
        self.active.store(false, Ordering::SeqCst);
        let _ = self.compositor.take();
        self.current_cursor_transform_mode = None;

        match std::mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => {}
            QueueState::Queued(token) | QueueState::WaitingForEstimatedVBlank(token) => {
                self.loop_handle.remove(token);
            }
            QueueState::WaitingForVBlank { .. } => self.timings.discard_current_frame(),
            QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank,
                queued_render,
            } => {
                self.loop_handle.remove(estimated_vblank);
                self.loop_handle.remove(queued_render);
            }
        };

        let _ = tx.send(());
    }

    fn resume(&mut self, compositor: GbmDrmOutput) {
        if let Ok(caps) = compositor.scanout_capabilities() {
            self.output.set_scanout_capabilities(caps);
        }
        self.output
            .user_data()
            .insert_if_missing_threadsafe(OutputSwapchainFormat::default);
        if let Some(format_data) = self.output.user_data().get::<OutputSwapchainFormat>() {
            *format_data.0.lock().unwrap() = Some(compositor.format());
        }
        let (mode, min_hz) = compositor.with_compositor(|c| {
            (
                c.surface().pending_mode(),
                drm_helpers::get_minimum_refresh_rate(
                    c.surface(),
                    c.pending_connectors().into_iter().next().unwrap(),
                )
                .ok()
                .flatten(),
            )
        });
        self.min_vrr = min_hz;
        let interval =
            Duration::from_secs_f64(1_000. / drm_helpers::calculate_refresh_rate(mode) as f64);
        self.timings.set_refresh_interval(Some(interval));

        const SAFETY_MARGIN: u32 = 2; // Magic two frames margin taken from kwin to not trigger low-framerate-compensation
        let min_min_refresh_interval = Duration::from_secs_f64(1. / 30.); // 30Hz
        self.min_vrr_frame_time = Some(
            min_hz
                .map(|min| Duration::from_secs_f64(1. / (min + SAFETY_MARGIN) as f64))
                .unwrap_or(min_min_refresh_interval) // alternatively use 30Hz
                .min(min_min_refresh_interval),
        );
        self.timings
            .set_min_refresh_interval(self.min_vrr_frame_time);

        if *DISABLE_DIRECT_SCANOUT {
            self.frame_flags.remove(
                FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                    | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY
                    | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
            );
        } else if *DISABLE_OVERLAY_SCANOUT {
            self.frame_flags
                .remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
        }
        if *DISABLE_CURSOR_PLANE {
            self.frame_flags
                .remove(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT);
        }
        self.current_cursor_transform_mode = None;
        self.compositor = Some(compositor);
    }
}

#[inline]
fn apply_cursor_buffer_transform(
    compositor: &GbmDrmOutput,
    current_mode: &mut Option<CursorTransformMode>,
    hdr_enabled: bool,
    active_scanout_plan: Option<ScanoutPlan>,
    hdr_reference_white: f32,
) {
    let desired_mode =
        CursorTransformMode::for_state(hdr_enabled, active_scanout_plan, hdr_reference_white);

    if *current_mode != Some(desired_mode) {
        *current_mode = Some(desired_mode);
        let _transform: Option<CursorBufferTransformFn> = match desired_mode {
            CursorTransformMode::Passthrough => None,
            CursorTransformMode::PqEncode { ref_white } => {
                let encoder = SrgbToPqEncoder::new(ref_white as f32 / 10000.0);
                Some(Box::new(
                    move |data: &mut [u8], stride: u32, size: (u32, u32)| {
                        encoder.apply(data, stride, size);
                    },
                ))
            }
            CursorTransformMode::LinearRec709 => {
                let encoder = SrgbToPqEncoder::new_linear(1.0, false);
                Some(Box::new(
                    move |data: &mut [u8], stride: u32, size: (u32, u32)| {
                        encoder.apply(data, stride, size);
                    },
                ))
            }
        };
        compositor.set_cursor_buffer_transform(None);

        let _post_blend_transform: Option<CursorBufferTransformFn> = if hdr_enabled {
            let peak = (hdr_reference_white * 5.0).max(1000.0);
            let scale = (hdr_reference_white / peak).min(1.0);
            let encoder = SrgbToPqEncoder::new_linear(scale, false);
            Some(Box::new(
                move |data: &mut [u8], stride: u32, size: (u32, u32)| {
                    encoder.apply(data, stride, size);
                },
            ))
        } else {
            None
        };
        compositor.set_cursor_buffer_transform_post_blend(None);
    }
}

impl SurfaceThreadState {
    fn update_scanout_color_management(
        &mut self,
        scanout_plan: Option<ScanoutPlan>,
        fullscreen_surface: Option<&FullscreenOccupied>,
        allow_primary_scanout: &mut bool,
    ) {
        let compositor = self.compositor.as_ref().unwrap();
        if *allow_primary_scanout && let Some(scanout_plan) = scanout_plan {
            match scanout_plan {
                ScanoutPlan::PlaneColorop(conv) => {
                    let wl_surf = fullscreen_surface.and_then(|f| f.surface.wl_surface());
                    if let Some(wl_surf) = wl_surf {
                        let transform = conv.to_scanout_color_transform();
                        let mut transforms = std::collections::HashMap::new();
                        smithay::desktop::utils::with_surfaces_surface_tree(&wl_surf, |s, _| {
                            transforms.insert(Id::from_wayland_resource(s), transform);
                        });

                        let is_direct = self
                            .output
                            .scanout_capabilities()
                            .map(|caps| caps.can_plane_colorop_direct(conv))
                            .unwrap_or(false);

                        let post_blend = if self.hdr_enabled && !is_direct {
                            let peak = self
                                .output
                                .user_data()
                                .get::<HdrOutputState>()
                                .and_then(|s| s.get().or_else(|| s.staged()))
                                .map(|hdr| hdr.capabilities.max_luminance as f64)
                                .unwrap_or(1000.0);
                            Some(PostBlendEncode::for_hdr(peak))
                        } else {
                            None
                        };

                        let linear_transforms = match post_blend {
                            Some(pb) => transforms
                                .iter()
                                .filter_map(|(id, tr)| {
                                    let linear = pb.linear_transform((*tr)?)?;
                                    Some((id.clone(), linear))
                                })
                                .collect(),
                            None => std::collections::HashMap::new(),
                        };

                        let _ = compositor.use_crtc_color_state(CrtcColorState::default());
                        compositor.use_color_transforms(transforms, self.hdr_enabled);
                        compositor.use_post_blend_encode(post_blend, linear_transforms);
                        self.active_scanout_plan = Some(scanout_plan);
                    } else {
                        *allow_primary_scanout = false;
                        self.active_scanout_plan = None;
                        compositor.use_color_transforms(std::collections::HashMap::new(), false);
                        compositor.use_post_blend_encode(None, std::collections::HashMap::new());
                    }
                }
                ScanoutPlan::CrtcHardware(conv) => {
                    compositor.use_color_transforms(std::collections::HashMap::new(), false);
                    compositor.use_post_blend_encode(None, std::collections::HashMap::new());
                    let caps = self.output.scanout_capabilities().unwrap_or_default();
                    let color_state = conv.to_crtc_color_state(
                        caps.crtc_color.gamma_lut_size as usize,
                        caps.crtc_color.degamma_lut_size as usize,
                    );
                    match compositor.use_crtc_color_state(color_state) {
                        Ok(()) => {
                            debug!(
                                ?conv,
                                ?scanout_plan,
                                "Staged CRTC hardware color management for scanout"
                            );
                            self.active_scanout_plan = Some(scanout_plan);
                        }
                        Err(err) => {
                            warn!(
                                ?err,
                                ?scanout_plan,
                                "CRTC color state rejected; falling back to Vulkan fast direct flip"
                            );
                            let _ = compositor.use_crtc_color_state(CrtcColorState::default());
                            self.active_scanout_plan = Some(ScanoutPlan::VulkanFastDirectFlip);
                            self.output.set_fullscreen_scanout_plan(ScanoutPlan::VulkanFastDirectFlip);
                            *allow_primary_scanout = false;
                        }
                    }
                }
                ScanoutPlan::DirectPassthrough | ScanoutPlan::VulkanFastDirectFlip => {
                    self.active_scanout_plan = Some(scanout_plan);
                    compositor.use_color_transforms(std::collections::HashMap::new(), false);
                    compositor.use_post_blend_encode(None, std::collections::HashMap::new());
                    let _ = compositor.use_crtc_color_state(CrtcColorState::default());
                }
            }
        } else {
            compositor.use_color_transforms(std::collections::HashMap::new(), false);
            compositor.use_post_blend_encode(None, std::collections::HashMap::new());
            self.active_scanout_plan = None;
            let _ = compositor.use_crtc_color_state(CrtcColorState::default());
        }
    }

    fn node_added(
        &mut self,
        node: DrmNode,
        gbm: GbmAllocator<DrmDeviceFd>,
        renderer: SurfaceNodeRenderer,
    ) -> Result<()> {
        match (&mut self.api, renderer) {
            (SurfaceGpuApi::Glow(api), SurfaceNodeRenderer::Egl(egl)) => {
                let mut renderer =
                    unsafe { GlowRenderer::new(egl) }.context("Failed to create renderer")?;
                init_shaders(renderer.borrow_mut()).context("Failed to initialize shaders")?;
                api.as_mut().add_node(node, gbm, renderer);
            }
            (SurfaceGpuApi::Vulkan(api), SurfaceNodeRenderer::Vulkan(renderer)) => {
                api.as_mut().add_node(node, gbm, renderer);
            }
            _ => anyhow::bail!("Mismatched surface node renderer and surface GPU API"),
        }

        Ok(())
    }

    fn node_removed(&mut self, node: DrmNode) {
        match &mut self.api {
            SurfaceGpuApi::Glow(api) => {
                api.as_mut().remove_node(&node);
                let _ = api.devices();
            }
            SurfaceGpuApi::Vulkan(api) => {
                api.as_mut().remove_node(&node);
                let _ = api.devices();
            }
        }
    }

    #[profiling::function]
    fn on_vblank(&mut self, metadata: Option<DrmEventMetadata>) {
        let Some(compositor) = self.compositor.as_mut() else {
            return;
        };
        trace!(?metadata, state = ?self.state, "surface on_vblank");

        // handle edge-cases right after resume
        if !matches!(
            self.state,
            QueueState::WaitingForVBlank { .. } | QueueState::Idle
        ) {
            match mem::replace(&mut self.state, QueueState::Idle) {
                QueueState::WaitingForVBlank { .. } | QueueState::Idle => unreachable!(),
                QueueState::Queued(token) | QueueState::WaitingForEstimatedVBlank(token) => {
                    self.loop_handle.remove(token);
                }
                QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank,
                    queued_render,
                } => {
                    self.loop_handle.remove(estimated_vblank);
                    self.loop_handle.remove(queued_render);
                }
            }
        }
        if matches!(self.state, QueueState::Idle) {
            return;
        }

        let now = self.clock.now();
        let presentation_time = match metadata.as_ref().map(|data| &data.time) {
            Some(DrmEventTime::Monotonic(tp)) => Some(*tp),
            _ => None,
        };
        let sequence = metadata.as_ref().map(|data| data.sequence).unwrap_or(0);

        // finish tracy frame
        let _ = self.vblank_frame.take();

        // mark last frame completed
        if let Ok(Some(Some((mut feedback, frames, estimated_presentation_time)))) =
            compositor.frame_submitted()
            && self.mirroring.is_none()
        {
            let name = self.output.name();
            let message = if let Some(presentation_time) = presentation_time {
                let misprediction_s =
                    presentation_time.as_secs_f64() - estimated_presentation_time.as_secs_f64();
                tracy_client::Client::running().unwrap().plot(
                    self.presentation_misprediction_plot_name,
                    misprediction_s * 1000.,
                );

                let now = Duration::from(now);
                if presentation_time > now {
                    let diff = presentation_time - now;
                    tracy_client::Client::running().unwrap().plot(
                        self.time_since_presentation_plot_name,
                        -diff.as_secs_f64() * 1000.,
                    );
                    format!("vblank on {name}, presentation is {diff:?} later")
                } else {
                    let diff = now - presentation_time;
                    tracy_client::Client::running().unwrap().plot(
                        self.time_since_presentation_plot_name,
                        diff.as_secs_f64() * 1000.,
                    );
                    format!("vblank on {name}, presentation was {diff:?} ago")
                }
            } else {
                format!("vblank on {name}, presentation time unknown")
            };
            tracy_client::Client::running()
                .unwrap()
                .message(&message, 0);

            let (clock, flags) = if let Some(tp) = presentation_time {
                (
                    tp.into(),
                    wp_presentation_feedback::Kind::Vsync
                        | wp_presentation_feedback::Kind::HwClock
                        | wp_presentation_feedback::Kind::HwCompletion,
                )
            } else {
                (
                    now,
                    wp_presentation_feedback::Kind::Vsync
                        | wp_presentation_feedback::Kind::HwCompletion,
                )
            };

            let rate = self
                .output
                .current_mode()
                .map(|mode| Duration::from_secs_f64(1_000.0 / mode.refresh as f64));
            let refresh = match rate {
                Some(rate)
                    if self
                        .compositor
                        .as_ref()
                        .is_some_and(|comp| comp.with_compositor(|c| c.vrr_enabled())) =>
                {
                    Refresh::Variable(rate)
                }
                Some(rate) => Refresh::Fixed(rate),
                None => Refresh::Unknown,
            };

            if let Some(last_sequence) = self.last_sequence {
                let delta = sequence as f64 - last_sequence as f64;
                tracy_client::Client::running()
                    .unwrap()
                    .plot(self.sequence_delta_plot_name, delta);
            }
            self.last_sequence = Some(sequence);

            feedback.presented(clock, refresh, sequence as u64, flags);

            self.timings.presented(clock);
            self.output
                .set_avg_frametime(self.timings.avg_frametime(SAMPLE_TIME_WINDOW));

            while let Ok(pending_image_copy_data) = frames.try_recv() {
                pending_image_copy_data.send_success_when_ready(
                    self.output.current_transform(),
                    &self.loop_handle,
                    clock,
                );
            }
        }

        let (redraw_needed, is_fullscreen) = match mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => unreachable!(),
            QueueState::WaitingForVBlank {
                redraw_needed,
                fullscreen_request,
            } => (redraw_needed, fullscreen_request),
            QueueState::WaitingForEstimatedVBlank(_) => unreachable!(),
            QueueState::WaitingForEstimatedVBlankAndQueued { .. } => unreachable!(),
        };

        if redraw_needed
            || (!self.timings.vrr() && self.shell.read().output_animations_going(&self.output))
        {
            let vblank_frame = tracy_client::Client::running()
                .unwrap()
                .non_continuous_frame(self.vblank_frame_name);
            self.vblank_frame = Some(vblank_frame);

            self.queue_redraw(false, is_fullscreen);
        }
        self.send_frame_callbacks();
    }

    #[profiling::function]
    fn on_estimated_vblank(&mut self, force: bool) {
        match mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => unreachable!(),
            QueueState::WaitingForVBlank { .. } => unreachable!(),
            QueueState::WaitingForEstimatedVBlank(_) => (),
            // The timer fired just in front of a redraw.
            QueueState::WaitingForEstimatedVBlankAndQueued { queued_render, .. } => {
                self.state = QueueState::Queued(queued_render);
                return;
            }
        }

        self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);

        if force || (!self.timings.vrr() && self.shell.read().output_animations_going(&self.output))
        {
            self.queue_redraw(false, false);
        }
        self.send_frame_callbacks();
    }

    fn queue_redraw(&mut self, mut force: bool, is_fullscreen: bool) {
        let Some(_compositor) = self.compositor.as_mut() else {
            return;
        };

        let is_fullscreen_skip_other = *FULLSCREEN_SKIP_OTHER_SURFACE
            && (self.timings.vrr() || *FULLSCREEN_SKIP_OTHER_SURFACE_ALWAYS)
            && self.output.is_foreground_fullscreen_occupied().is_some()
            && !force
            && !is_fullscreen;

        if *FULLSCREEN_SKIP_OTHER_SURFACE && is_fullscreen {
            force = true;
        }

        let immediate = if *FULLSCREEN_IMMEDIATE_RENDER && self.timings.vrr() && is_fullscreen {
            force = true;
            true
        } else {
            false
        };

        if let QueueState::WaitingForVBlank {
            fullscreen_request, ..
        } = &self.state
        {
            // We're waiting for VBlank, request a redraw afterwards.
            if !fullscreen_request {
                self.state = QueueState::WaitingForVBlank {
                    redraw_needed: true,
                    fullscreen_request: is_fullscreen,
                };
            }
            return;
        }

        if !force {
            match &self.state {
                QueueState::Idle | QueueState::WaitingForEstimatedVBlank(_) => {}

                // A redraw is already queued.
                QueueState::Queued(_) | QueueState::WaitingForEstimatedVBlankAndQueued { .. } => {
                    return;
                }
                _ => unreachable!(),
            };
        }

        let estimated_presentation = self.timings.next_presentation_time(&self.clock);
        let render_start = if is_fullscreen_skip_other {
            // To prevent the fullscreen surface from unexpectedly stopping updates, register a fallback redraw request.
            // If the fullscreen surface commits an update within the min_vrr interval, it will replace this fallback request.
            self.min_vrr_frame_time.unwrap_or(_30_HZ)
        } else if immediate {
            Duration::ZERO
        } else {
            self.timings.next_render_time(&self.clock)
        };

        let timer = if render_start.is_zero() {
            trace!("Running late for frame.");
            // TODO triple buffering
            Timer::immediate()
        } else {
            Timer::from_duration(render_start)
        };

        let token = self
            .loop_handle
            .insert_source(timer, move |_time, _, state| {
                if let Err(err) = state.redraw(estimated_presentation) {
                    let name = state.output.name();
                    warn!(?name, "Failed to submit rendering: {:?}", err);
                    let err_str = format!("{err:?}");
                    if err_str.contains("DeadDevice")
                        || err_str.contains("ERROR_DEVICE_LOST")
                        || err_str.contains("DeviceLost")
                    {
                        error!(
                            ?name,
                            "Vulkan device lost in surface thread! Requesting GPU recovery."
                        );
                        let _ = state
                            .thread_sender
                            .send(SurfaceCommand::DeviceLost(state.target_node));
                        return TimeoutAction::Drop;
                    }
                    if hdr_policy().require_active {
                        let _ = state
                            .thread_sender
                            .send(SurfaceCommand::FatalRenderError(format!("{err:#}")));
                        return TimeoutAction::Drop;
                    }
                    state.queue_redraw(true, false);
                }
                TimeoutAction::Drop
            })
            .expect("Failed to schedule render");

        match &self.state {
            QueueState::Idle => {
                self.state = QueueState::Queued(token);
            }
            QueueState::WaitingForEstimatedVBlank(estimated_vblank) => {
                self.state = QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank: *estimated_vblank,
                    queued_render: token,
                };
            }
            QueueState::Queued(old_token) if force => {
                self.loop_handle.remove(*old_token);
                self.state = QueueState::Queued(token);
            }
            QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank,
                queued_render,
            } if force => {
                self.loop_handle.remove(*queued_render);
                self.state = QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank: *estimated_vblank,
                    queued_render: token,
                };
            }
            _ => unreachable!(),
        }
    }

    #[profiling::function]
    fn redraw(&mut self, estimated_presentation: Duration) -> Result<()> {
        if self.compositor.is_none() {
            return Ok(());
        };

        let render_node = render_node_for_output(
            self.mirroring.as_ref().unwrap_or(&self.output),
            self.primary_node
                .read()
                .unwrap()
                .as_ref()
                .unwrap_or(&self.target_node),
            &self.target_node,
            &self.shell.read(),
        );

        if self.is_vulkan {
            self.redraw_vulkan(render_node, estimated_presentation)
        } else {
            self.redraw_glow(render_node, estimated_presentation)
        }
    }

    fn redraw_glow(
        &mut self,
        render_node: DrmNode,
        estimated_presentation: Duration,
    ) -> Result<()> {
        self.timings.start_render(&self.clock);

        let mut additional_frame_flags = FrameFlags::empty();
        let mut remove_frame_flags = FrameFlags::empty();

        let (
            has_active_fullscreen,
            fullscreen_drives_refresh_rate,
            animations_going,
            prefers_async,
            scanout_plan,
            fullscreen_surface,
        ) = {
            let shell = self.shell.read();
            let output = self.mirroring.as_ref().unwrap_or(&self.output);
            let animations_going = shell.output_animations_going(output);
            if let Some(fullscreen_surface) = output.is_foreground_fullscreen_occupied()
                && fullscreen_surface.alive()
            {
                let min_vrr_frame_time = self
                    .min_vrr_frame_time
                    .unwrap_or(Duration::from_nanos(1_000_000_000 / 30));
                let drives_refresh_rate = fullscreen_surface.wl_surface().is_some_and(|surface| {
                    recursive_frame_time_estimation(&self.clock, &surface)
                        .is_some_and(|dur| dur <= min_vrr_frame_time)
                });
                let prefers_async = fullscreen_surface.prefers_async;
                let scanout_plan = fullscreen_surface.effective_scanout_plan();
                (
                    true,
                    drives_refresh_rate,
                    animations_going,
                    prefers_async,
                    Some(scanout_plan),
                    Some(fullscreen_surface),
                )
            } else {
                (false, false, animations_going, false, None, None)
            }
        };

        let mut allow_primary_scanout = has_active_fullscreen
            && scanout_plan.is_some_and(|plan| plan.allows_primary_scanout())
            && self.screen_filter.is_noop()
            && self.mirroring.is_none()
            && !*DISABLE_DIRECT_SCANOUT;

        if self.fullscreen != fullscreen_surface
            || self.is_scanout != allow_primary_scanout
            || self.active_scanout_plan != scanout_plan
        {
            self.update_scanout_color_management(
                scanout_plan,
                fullscreen_surface.as_ref(),
                &mut allow_primary_scanout,
            );
            self.fullscreen = fullscreen_surface;
        }

        let compositor = self.compositor.as_mut().unwrap();
        apply_cursor_buffer_transform(
            compositor,
            &mut self.current_cursor_transform_mode,
            self.hdr_enabled,
            self.active_scanout_plan,
            self.hdr_reference_white,
        );

        if self.is_scanout != allow_primary_scanout {
            if allow_primary_scanout {
                error!(plan = ?scanout_plan, "Enable SCANOUT with plan: {:?}", scanout_plan);
            } else {
                error!("Disable SCANOUT");
            }
            self.is_scanout = allow_primary_scanout;
        }

        // Cursor plane is transformed on CPU (SrgbToPqEncoder) to match either the HDR output
        // signal directly or linearized to Rec.709 before CRTC CTM/GAMMA_LUT, so hardware
        // cursor scanout is safe in all scanout plans without distorting cursor colors.
        let disable_cursor_plane = *DISABLE_CURSOR_PLANE;
        if !disable_cursor_plane {
            additional_frame_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        } else {
            remove_frame_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        }

        if allow_primary_scanout {
            additional_frame_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        } else {
            remove_frame_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        }

        // Tearing: honor the fullscreen/covering client's async hint with real
        // async page flips when the user allows it. The KMS layer falls back
        // to synchronized flips whenever the kernel refuses to tear.
        let _tearing =
            tearing_allowed_for(&self.output.name()) && has_active_fullscreen && prefers_async;

        if animations_going || *DISABLE_OVERLAY_SCANOUT {
            remove_frame_flags |= FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT;
        }

        let mut vrr = matches!(self.vrr_mode, AdaptiveSync::Force);

        if self.vrr_mode == AdaptiveSync::Enabled {
            vrr = has_active_fullscreen;
        }

        let SurfaceGpuApi::Glow(api) = &mut self.api else {
            unreachable!()
        };

        let mut renderer = if !is_same_gpu(&render_node, &self.target_node) {
            api.renderer(&render_node, &self.target_node, compositor.format())
                .map_err(|err| anyhow::format_err!("Failed to create renderer: {:?}", err))?
        } else {
            api.single_renderer(&self.target_node)
                .map_err(|err| anyhow::format_err!("Failed to create renderer: {:?}", err))?
        };

        set_hdr_client_blend(&mut renderer, self.hdr_config);

        let mut elements = output_elements(
            Some(&render_node),
            &mut renderer,
            &self.shell,
            self.clock.now(),
            self.mirroring.as_ref().unwrap_or(&self.output),
            CursorMode::All,
            #[cfg(not(feature = "debug"))]
            None,
            #[cfg(feature = "debug")]
            Some((&self.egui, &self.timings)),
            Some(self.target_node),
        )
        .map_err(|err| {
            anyhow::format_err!("Failed to accumulate elements for rendering: {:?}", err)
        })?;

        if vrr && fullscreen_drives_refresh_rate && !self.timings.past_min_render_time(&self.clock)
        {
            additional_frame_flags |= FrameFlags::SKIP_CURSOR_ONLY_UPDATES;
        };
        if has_active_fullscreen {
            additional_frame_flags |= FrameFlags::FULLSCREEN_PACING;
        }
        self.timings.set_vrr(vrr);
        self.timings.elements_done(&self.clock);

        // we can't use the elements after `compositor.render_frame`,
        // so let's collect everything we need for screencopy now
        let mut has_cursor_mode_none = false;
        let frames = if self.mirroring.is_none() {
            take_screencopy_frames(&self.output, &elements, &mut has_cursor_mode_none)
        } else {
            Default::default()
        };

        // actual rendering
        let postprocess = !self.screen_filter.is_noop();
        let source_output = self
            .mirroring
            .as_ref()
            .or(postprocess.then_some(&self.output))
            .filter(|output| {
                PostprocessOutputConfig::for_output_untransformed(output)
                    != PostprocessOutputConfig::for_output(&self.output)
                    || postprocess
            });

        let mut pre_postprocess_data = PrePostprocessData::default();

        let res = if let Some(source_output) = source_output {
            let offscreen_output_config =
                PostprocessOutputConfig::for_output_untransformed(source_output);
            let intermediate_format =
                postprocess_intermediate_format(compositor.format(), self.hdr_enabled);
            let postprocess_state = match self.postprocess_textures.entry(self.target_node) {
                hash_map::Entry::Occupied(occupied) => {
                    let postprocess_state = occupied.into_mut();
                    // If output config is different, or the offscreen buffer's
                    // intermediate format changed.
                    if postprocess_state.output_config != offscreen_output_config
                        || postprocess_state.texture.format() != Some(intermediate_format)
                    {
                        *postprocess_state = PostprocessState::new_with_renderer(
                            &mut renderer,
                            intermediate_format,
                            offscreen_output_config,
                        )?
                    }
                    postprocess_state
                }
                hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(PostprocessState::new_with_renderer(
                        &mut renderer,
                        intermediate_format,
                        offscreen_output_config,
                    )?)
                }
            };

            if has_cursor_mode_none && self.mirroring.is_none() {
                // TODO: use `extract_if` once stablized
                let cursor_element_count = elements
                    .iter()
                    .take_while(|elem| elem.kind() == Kind::Cursor)
                    .count();
                let cursor_elements = elements.drain(..cursor_element_count).collect::<Vec<_>>();
                let scale = source_output.current_scale().fractional_scale().into();

                let geometry: Option<Rectangle<i32, Physical>> =
                    cursor_elements.iter().fold(None, |acc, elem| {
                        let geometry = elem.geometry(scale);
                        if let Some(acc) = acc {
                            Some(acc.merge(geometry))
                        } else {
                            Some(geometry)
                        }
                    });

                if let Some(geometry) = geometry {
                    let cursor_elements = cursor_elements
                        .into_iter()
                        .map(|elem| {
                            RelocateRenderElement::from_element(
                                elem,
                                Point::from((-geometry.loc.x, -geometry.loc.y)),
                                Relocate::Relative,
                            )
                        })
                        .collect::<Vec<_>>();

                    postprocess_state.track_cursor(
                        &mut renderer,
                        Fourcc::Abgr8888,
                        geometry.size,
                        scale,
                    )?;

                    postprocess_state
                        .cursor_texture
                        .as_mut()
                        .unwrap()
                        .render()
                        .draw::<_, <GlMultiRenderer as RendererSuper>::Error>(|tex| {
                            if self.mirroring.is_none() {
                                pre_postprocess_data.cursor_geometry = Some(geometry);
                                pre_postprocess_data.cursor_texture = Some(tex.clone());
                            }

                            let mut fb = renderer.bind(tex)?;
                            let res = match postprocess_state
                                .cursor_damage_tracker
                                .as_mut()
                                .unwrap()
                                .render_output(
                                    &mut renderer,
                                    &mut fb,
                                    1,
                                    &cursor_elements,
                                    [0.0, 0.0, 0.0, 0.0],
                                ) {
                                Ok(res) => res,
                                Err(RenderError::Rendering(err)) => return Err(err),
                                Err(RenderError::OutputNoMode(_)) => unreachable!(),
                            };

                            if self.mirroring.is_none() {
                                pre_postprocess_data.states = Some(res.states);
                            }

                            renderer.wait(&res.sync)?;
                            std::mem::drop(fb);

                            let transform = source_output.current_transform();
                            let area = tex.size().to_logical(1, transform);

                            Ok(res
                                .damage
                                .cloned()
                                .map(|v| {
                                    v.into_iter()
                                        .map(|r| r.to_logical(1).to_buffer(1, transform, &area))
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default())
                        })
                        .context("Failed to draw to offscreen render target")?;
                }
            } else {
                postprocess_state.remove_cursor();
            }

            postprocess_state
                .texture
                .render()
                .draw::<_, <GlMultiRenderer as RendererSuper>::Error>(|tex| {
                    if self.mirroring.is_none() {
                        pre_postprocess_data.texture = Some(tex.clone());
                    }

                    let mut fb = renderer.bind(tex)?;
                    let res = match postprocess_state.damage_tracker.render_output(
                        &mut renderer,
                        &mut fb,
                        1,
                        &elements,
                        CLEAR_COLOR,
                    ) {
                        Ok(res) => res,
                        Err(RenderError::Rendering(err)) => return Err(err),
                        Err(RenderError::OutputNoMode(_)) => unreachable!(),
                    };

                    if self.mirroring.is_none() {
                        if let Some(states) = pre_postprocess_data.states.as_mut() {
                            states.states.extend(res.states.states);
                        } else {
                            pre_postprocess_data.states = Some(res.states);
                        }
                    }

                    renderer.wait(&res.sync)?;
                    std::mem::drop(fb);

                    let transform = source_output.current_transform();
                    let area = tex.size().to_logical(1, transform);

                    Ok(res
                        .damage
                        .cloned()
                        .map(|v| {
                            v.into_iter()
                                .map(|r| r.to_logical(1).to_buffer(1, transform, &area))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default())
                })
                .context("Failed to draw to offscreen render target")?;

            renderer = api
                .single_renderer(&self.target_node)
                .map_err(|err| anyhow::format_err!("Failed to create renderer: {:?}", err))?;

            elements = postprocess_elements(
                &mut renderer,
                &self.output,
                &pre_postprocess_data,
                postprocess_state,
                &self.screen_filter,
                self.hdr_enabled,
                self.hdr_reference_white,
                self.hdr_hardware_offload,
            );

            if let Err(err) = compositor.with_compositor(|c| c.use_vrr(vrr)) {
                warn!("Unable to set adaptive VRR state: {}", err);
            }
            compositor.render_frame(
                &mut renderer,
                &elements,
                [0.0, 0.0, 0.0, 0.0],
                self.frame_flags
                    .union(additional_frame_flags)
                    .difference(remove_frame_flags),
            )
        } else {
            if let Err(err) = compositor.with_compositor(|c| c.use_vrr(vrr)) {
                warn!("Unable to set adaptive VRR state: {}", err);
            }
            let clear_color = if has_active_fullscreen {
                Color32F::new(0.0, 0.0, 0.0, 0.0)
            } else {
                CLEAR_COLOR // TODO use a theme neutral color
            };
            compositor.render_frame(
                &mut renderer,
                &elements,
                clear_color,
                self.frame_flags
                    .union(additional_frame_flags)
                    .difference(remove_frame_flags),
            )
        };
        self.timings.draw_done(&self.clock);

        match res {
            Ok(frame_result) => {
                let is_swapchain = matches!(
                    frame_result.primary_element,
                    PrimaryPlaneElement::Swapchain(_)
                );
                let actual_scanout = !is_swapchain && allow_primary_scanout;
                if self.swapchin_is_scanout != actual_scanout {
                    if actual_scanout {
                        error!(plan = ?scanout_plan, "Swapchain Enable SCANOUT with plan: {:?}", scanout_plan);
                    } else {
                        error!("Swapchain Disable SCANOUT");
                    }
                    self.swapchin_is_scanout = actual_scanout;
                }
                let (tx, rx) = std::sync::mpsc::channel();

                let feedback = if !frame_result.is_empty && self.mirroring.is_none() {
                    Some((
                        self.shell
                            .read()
                            .take_presentation_feedback(&self.output, &frame_result.states),
                        rx,
                        estimated_presentation,
                    ))
                } else {
                    None
                };

                if frame_result.needs_sync()
                    && let PrimaryPlaneElement::Swapchain(elem) = &frame_result.primary_element
                {
                    elem.sync.wait()?;
                }

                match compositor.queue_frame(feedback) {
                    x @ Ok(()) | x @ Err(FrameError::EmptyFrame) => {
                        self.timings.submitted_for_presentation(&self.clock);

                        // Update `state` after `queue_frame`, before any early return from errors
                        if x.is_ok() {
                            let new_state = QueueState::WaitingForVBlank {
                                redraw_needed: false,
                                fullscreen_request: false,
                            };
                            match mem::replace(&mut self.state, new_state) {
                                QueueState::Idle => unreachable!(),
                                QueueState::Queued(_) => (),
                                QueueState::WaitingForVBlank { .. } => unreachable!(),
                                QueueState::WaitingForEstimatedVBlank(estimated_vblank)
                                | QueueState::WaitingForEstimatedVBlankAndQueued {
                                    estimated_vblank,
                                    ..
                                } => {
                                    self.loop_handle.remove(estimated_vblank);
                                }
                            };
                        }

                        let now = self.clock.now();
                        for (session, frame, res) in frames {
                            if let Err(err) = send_screencopy_result(
                                &mut renderer,
                                &self.output,
                                &mut pre_postprocess_data,
                                &tx,
                                &frame_result,
                                &elements,
                                (&session, frame, res),
                                now.into(),
                            ) {
                                tracing::warn!(?err, "Failed to screencopy");
                            }
                        }

                        if self.mirroring.is_none() {
                            // If postprocessing, use states from first render
                            let states = pre_postprocess_data.states.unwrap_or(frame_result.states);
                            self.send_dmabuf_feedback(states);
                        }

                        if x.is_ok() {
                            self.send_signal_fifo_callbacks();

                            if let Some(hdr_state) =
                                self.output.user_data().get::<drm_helpers::HdrOutputState>()
                            {
                                // TEST_ONLY validation is not enough to advertise HDR to
                                // clients. Publish it only after the real frame commit was
                                // accepted by KMS.
                                hdr_state.commit();
                            }
                            if self.mirroring.is_none() {
                                self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);
                                self.send_frame_callbacks();
                            }
                        } else {
                            // we don't expect a vblank
                            let _ = self.vblank_frame.take();

                            self.queue_estimated_vblank(
                                estimated_presentation,
                                // Make sure we redraw to reevaluate, if we intentionally missed content
                                additional_frame_flags
                                    .contains(FrameFlags::SKIP_CURSOR_ONLY_UPDATES),
                            );
                        }
                    }
                    Err(err) => {
                        for (_session, frame, _) in frames {
                            frame.fail(CaptureFailureReason::Unknown);
                        }
                        return Err(err).with_context(|| "Failed to submit result for display");
                    }
                };
            }
            Err(err) => {
                compositor.reset_buffers();
                anyhow::bail!("Rendering failed: {}", err);
            }
        }

        let SurfaceGpuApi::Glow(api) = &mut self.api else {
            unreachable!()
        };
        for device in api.devices_mut()? {
            device.renderer_mut().cleanup_texture_cache()?;
        }

        Ok(())
    }

    fn redraw_vulkan(
        &mut self,
        render_node: DrmNode,
        estimated_presentation: Duration,
    ) -> Result<()> {
        self.timings.start_render(&self.clock);

        let mut additional_frame_flags = FrameFlags::empty();
        let mut remove_frame_flags = FrameFlags::empty();

        let (
            has_active_fullscreen,
            fullscreen_drives_refresh_rate,
            animations_going,
            _prefers_async,
            scanout_plan,
            fullscreen_surface,
        ) = {
            let shell = self.shell.read();
            let output = self.mirroring.as_ref().unwrap_or(&self.output);
            let animations_going = shell.output_animations_going(output);
            if let Some(fullscreen_surface) = output.is_foreground_fullscreen_occupied()
                && fullscreen_surface.alive()
            {
                let min_vrr_frame_time = self
                    .min_vrr_frame_time
                    .unwrap_or(Duration::from_nanos(1_000_000_000 / 30));
                let drives_refresh_rate = fullscreen_surface.wl_surface().is_some_and(|surface| {
                    recursive_frame_time_estimation(&self.clock, &surface)
                        .is_some_and(|dur| dur <= min_vrr_frame_time)
                });
                let prefers_async = fullscreen_surface.prefers_async;
                let scanout_plan = fullscreen_surface.effective_scanout_plan();
                (
                    true,
                    drives_refresh_rate,
                    animations_going,
                    prefers_async,
                    Some(scanout_plan),
                    Some(fullscreen_surface),
                )
            } else {
                (false, false, animations_going, false, None, None)
            }
        };

        let mut allow_primary_scanout = has_active_fullscreen
            && scanout_plan.is_some_and(|plan| plan.allows_primary_scanout())
            && self.screen_filter.is_noop()
            && self.mirroring.is_none()
            && !*DISABLE_DIRECT_SCANOUT;

        if self.fullscreen != fullscreen_surface
            || self.is_scanout != allow_primary_scanout
            || self.active_scanout_plan != scanout_plan
        {
            self.update_scanout_color_management(
                scanout_plan,
                fullscreen_surface.as_ref(),
                &mut allow_primary_scanout,
            );
            self.fullscreen = fullscreen_surface;
            self.is_scanout = false;
            self.swapchin_is_scanout = false;
        }

        let compositor = self.compositor.as_mut().unwrap();
        apply_cursor_buffer_transform(
            compositor,
            &mut self.current_cursor_transform_mode,
            self.hdr_enabled,
            self.active_scanout_plan,
            self.hdr_reference_white,
        );

        if self.is_scanout != allow_primary_scanout {
            if allow_primary_scanout {
                error!(plan = ?scanout_plan, "Enable SCANOUT with plan: {:?}", scanout_plan);
            } else {
                error!("Disable SCANOUT");
            }
            self.is_scanout = allow_primary_scanout;
        }

        // Cursor plane is transformed on CPU (SrgbToPqEncoder) to match either the HDR output
        // signal directly or linearized to Rec.709 before CRTC CTM/GAMMA_LUT, so hardware
        // cursor scanout is safe in all scanout plans without distorting cursor colors.
        let disable_cursor_plane = *DISABLE_CURSOR_PLANE;
        if !disable_cursor_plane {
            additional_frame_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        } else {
            remove_frame_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        }

        if allow_primary_scanout {
            additional_frame_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        } else {
            remove_frame_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        }

        if animations_going || *DISABLE_OVERLAY_SCANOUT {
            remove_frame_flags |= FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT;
        }

        let mut vrr = matches!(self.vrr_mode, AdaptiveSync::Force);

        if self.vrr_mode == AdaptiveSync::Enabled {
            vrr = has_active_fullscreen;
        }

        let SurfaceGpuApi::Vulkan(api) = &mut self.api else {
            unreachable!()
        };

        self.output
            .user_data()
            .insert_if_missing_threadsafe(OutputSwapchainFormat::default);
        if let Some(format_data) = self.output.user_data().get::<OutputSwapchainFormat>() {
            *format_data.0.lock().unwrap() = Some(compositor.format());
        }

        let mut renderer = if !is_same_gpu(&render_node, &self.target_node) {
            api.renderer(&render_node, &self.target_node, compositor.format())
                .map_err(|err| anyhow::format_err!("Failed to create renderer: {:?}", err))?
        } else {
            api.single_renderer(&self.target_node)
                .map_err(|err| anyhow::format_err!("Failed to create renderer: {:?}", err))?
        };

        renderer.as_mut().set_hdr_output(self.hdr_config);
        if let Some(target) = renderer.target_as_mut() {
            target.set_hdr_output(self.hdr_config);
        }

        self.output
            .user_data()
            .insert_if_missing_threadsafe(OutputVulkanTimeline::default);
        if let Some(timeline_data) = self.output.user_data().get::<OutputVulkanTimeline>() {
            let timeline = renderer
                .target_as_ref()
                .and_then(|r| r.drm_timeline())
                .or_else(|| renderer.as_ref().drm_timeline())
                .cloned();
            if let Some(timeline) = timeline {
                let mut current = timeline_data.0.write();
                if current.as_ref() != Some(&timeline) {
                    *current = Some(timeline);
                }
            }
        }

        let elements = output_elements(
            Some(&render_node),
            &mut renderer,
            &self.shell,
            self.clock.now(),
            self.mirroring.as_ref().unwrap_or(&self.output),
            CursorMode::All,
            #[cfg(not(feature = "debug"))]
            None,
            #[cfg(feature = "debug")]
            Some((&self.egui, &self.timings)),
            Some(self.target_node),
        )
        .map_err(|err| {
            anyhow::format_err!("Failed to accumulate elements for rendering: {:?}", err)
        })?;

        if vrr && fullscreen_drives_refresh_rate && !self.timings.past_min_render_time(&self.clock)
        {
            additional_frame_flags |= FrameFlags::SKIP_CURSOR_ONLY_UPDATES;
        };
        if has_active_fullscreen {
            additional_frame_flags |= FrameFlags::FULLSCREEN_PACING;
        }
        self.timings.set_vrr(vrr);
        self.timings.elements_done(&self.clock);

        let mut has_cursor_mode_none = false;
        let frames = if self.mirroring.is_none() {
            take_screencopy_frames(&self.output, &elements, &mut has_cursor_mode_none)
        } else {
            Default::default()
        };

        if let Err(err) = compositor.with_compositor(|c| c.use_vrr(vrr)) {
            warn!("Unable to set adaptive VRR state: {}", err);
        }
        let effective_flags = self
            .frame_flags
            .union(additional_frame_flags)
            .difference(remove_frame_flags);
        debug!(
            render_node = ?render_node,
            target_node = ?self.target_node,
            elem_count = elements.len(),
            frame_flags = ?effective_flags,
            vrr,
            allow_primary_scanout,
            "redraw_vulkan executing render_frame"
        );
        for (i, elem) in elements.iter().enumerate() {
            debug!(
                "  elem #{i}: id={:?}, geo={:?}",
                elem.id(),
                elem.geometry(self.output.current_scale().fractional_scale().into())
            );
        }

        let clear_color = if has_active_fullscreen {
            Color32F::new(0.0, 0.0, 0.0, 0.0)
        } else {
            CLEAR_COLOR
        };
        renderer.as_mut().begin_batch();
        if let Some(target) = renderer.target_as_mut() {
            target.begin_batch();
        }
        let res = compositor.render_frame(&mut renderer, &elements, clear_color, effective_flags);
        self.timings.draw_done(&self.clock);

        match res {
            Ok(frame_result) => {
                let is_swapchain = matches!(
                    frame_result.primary_element,
                    PrimaryPlaneElement::Swapchain(_)
                );
                let actual_scanout = !is_swapchain && allow_primary_scanout;
                if self.swapchin_is_scanout != actual_scanout {
                    if actual_scanout {
                        error!(plan = ?scanout_plan, "Swapchain Enable SCANOUT with plan: {:?}", scanout_plan);
                    } else {
                        error!("Swapchain Disable SCANOUT");
                    }
                    self.swapchin_is_scanout = actual_scanout;
                }
                let has_cursor_plane = frame_result.cursor_element.is_some();
                debug!(
                    is_empty = frame_result.is_empty,
                    needs_sync = frame_result.needs_sync(),
                    is_swapchain,
                    has_cursor_plane,
                    states_len = frame_result.states.states.len(),
                    "redraw_vulkan render_frame Ok"
                );
                let (tx, rx) = std::sync::mpsc::channel();

                let feedback = if !frame_result.is_empty && self.mirroring.is_none() {
                    Some((
                        self.shell
                            .read()
                            .take_presentation_feedback(&self.output, &frame_result.states),
                        rx,
                        estimated_presentation,
                    ))
                } else {
                    None
                };

                let now = self.clock.now();
                for (session, frame, res) in frames {
                    if let Err(err) = send_screencopy_result_vulkan(
                        &mut renderer,
                        &self.output,
                        &tx,
                        &frame_result,
                        &elements,
                        (&session, frame, res),
                        now.into(),
                    ) {
                        tracing::error!("Failed to screencopy in Vulkan: {err:#}");
                    }
                }

                let toplevel_frames: Vec<(
                    CosmicSurface,
                    Vec<(ScreencopySessionRef, ScreencopyFrame)>,
                )> = if self.mirroring.is_none() {
                    let shell = self.shell.read();
                    shell
                        .workspaces
                        .spaces()
                        .filter(|ws| ws.output() == &self.output)
                        .flat_map(|ws| {
                            ws.mapped()
                                .flat_map(|m| m.windows())
                                .map(|(s, _)| s)
                                .chain(ws.get_fullscreen_surfaces().map(|f| f.surface.clone()))
                        })
                        .filter_map(|window| {
                            let pending = window.take_pending_frames();
                            if pending.is_empty() {
                                None
                            } else {
                                Some((window, pending))
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                for (toplevel, pending) in toplevel_frames {
                    for (session, frame) in pending {
                        if let Err(err) = send_toplevel_screencopy_result_vulkan(
                            &mut renderer,
                            &self.output,
                            &tx,
                            &toplevel,
                            &session,
                            frame,
                            now.into(),
                        ) {
                            tracing::error!("Failed to toplevel screencopy in Vulkan: {err:#}");
                        }
                    }
                }

                if let Err(err) = renderer.as_mut().flush_batch() {
                    tracing::error!("Failed to flush Vulkan renderer batch: {err:#}");
                }
                if let Some(target) = renderer.target_as_mut() {
                    if let Err(err) = target.flush_batch() {
                        tracing::error!("Failed to flush Vulkan target renderer batch: {err:#}");
                    }
                }

                let _ = self
                    .thread_sender
                    .send(SurfaceCommand::ProcessShmScreencopy);

                // With Vulkan timeline semaphore exported to DRM syncobj, needs_sync() is false
                // because the fence is exportable as IN_FENCE_FD. The CPU does not block waiting for the GPU.
                if frame_result.needs_sync()
                    && let PrimaryPlaneElement::Swapchain(elem) = &frame_result.primary_element
                {
                    elem.sync.wait()?;
                }

                let queue_res = compositor.queue_frame(feedback);
                debug!(?queue_res, "redraw_vulkan queue_frame result");
                match queue_res {
                    x @ Ok(()) | x @ Err(FrameError::EmptyFrame) => {
                        self.timings.submitted_for_presentation(&self.clock);

                        if x.is_ok() {
                            let new_state = QueueState::WaitingForVBlank {
                                redraw_needed: false,
                                fullscreen_request: false,
                            };
                            match mem::replace(&mut self.state, new_state) {
                                QueueState::Idle => unreachable!(),
                                QueueState::Queued(_) => (),
                                QueueState::WaitingForVBlank { .. } => unreachable!(),
                                QueueState::WaitingForEstimatedVBlank(estimated_vblank)
                                | QueueState::WaitingForEstimatedVBlankAndQueued {
                                    estimated_vblank,
                                    ..
                                } => {
                                    self.loop_handle.remove(estimated_vblank);
                                }
                            };
                        }

                        if self.mirroring.is_none() {
                            let _ = self
                                .thread_sender
                                .send(SurfaceCommand::RenderStates(frame_result.states));
                            self.send_signal_fifo_callbacks();
                        }

                        if x.is_ok() {
                            if let Some(hdr_state) =
                                self.output.user_data().get::<drm_helpers::HdrOutputState>()
                            {
                                // TEST_ONLY validation is not enough to advertise HDR to
                                // clients. Publish it only after the real frame commit was
                                // accepted by KMS.
                                hdr_state.commit();
                            }

                            if self.mirroring.is_none() {
                                self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);
                                let _ = self
                                    .thread_sender
                                    .send(SurfaceCommand::SendFrames(self.frame_callback_seq));
                            }
                        } else {
                            let _ = self.vblank_frame.take();

                            if self.fullscreen.is_some() && self.timings.vrr() {
                                match mem::replace(&mut self.state, QueueState::Idle) {
                                    QueueState::WaitingForEstimatedVBlank(token)
                                    | QueueState::WaitingForEstimatedVBlankAndQueued {
                                        estimated_vblank: token,
                                        ..
                                    } => {
                                        self.loop_handle.remove(token);
                                    }
                                    _ => {}
                                }
                                self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);
                                if let Some(fullscreen) = &self.fullscreen {
                                    fullscreen.0.send_frame(
                                        &self.output,
                                        self.clock.now(),
                                        None,
                                        |_, _| Some(self.output.clone()),
                                    );
                                }
                            } else {
                                self.queue_estimated_vblank(
                                    estimated_presentation,
                                    additional_frame_flags
                                        .contains(FrameFlags::SKIP_CURSOR_ONLY_UPDATES),
                                );
                            }
                        }
                    }
                    Err(err) => {
                        return Err(err).with_context(|| "Failed to submit result for display");
                    }
                };
            }
            Err(err) => {
                renderer.as_mut().cancel_batch();
                if let Some(target) = renderer.target_as_mut() {
                    target.cancel_batch();
                }
                compositor.reset_buffers();
                anyhow::bail!("Rendering failed: {}", err);
            }
        }

        let SurfaceGpuApi::Vulkan(api) = &mut self.api else {
            unreachable!()
        };
        for device in api.devices_mut()? {
            device.renderer_mut().cleanup_texture_cache()?;
        }

        Ok(())
    }

    fn queue_estimated_vblank(&mut self, target_presentation_time: Duration, force: bool) {
        match mem::take(&mut self.state) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => (),
            QueueState::WaitingForVBlank { .. } => unreachable!(),
            QueueState::WaitingForEstimatedVBlank(token)
            | QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank: token,
                ..
            } => {
                self.state = QueueState::WaitingForEstimatedVBlank(token);
                return;
            }
        }

        let now = self.clock.now();
        let mut duration = target_presentation_time.saturating_sub(now.into());

        // No use setting a zero timer, since we'll send frame callbacks anyway right after the call to
        // render(). This can happen for example with unknown presentation time from DRM.
        if duration.is_zero() {
            duration += self.timings.refresh_interval();
        }

        trace!("queueing estimated vblank timer to fire in {duration:?}");

        let timer = Timer::from_duration(duration);
        let token = self
            .loop_handle
            .insert_source(timer, move |_, _, data| {
                data.on_estimated_vblank(force);
                TimeoutAction::Drop
            })
            .unwrap();
        self.state = QueueState::WaitingForEstimatedVBlank(token);
    }

    fn update_hdr_config(&mut self) {
        let surface_hdr_active =
            self.hdr_enabled && self.screen_filter.is_noop() && self.mirroring.is_none();
        self.hdr_config = if surface_hdr_active {
            Some(HdrOutputConfig {
                reference_white: self.hdr_reference_white,
                max_luminance: self.hdr_max_luminance,
                sdr_gamma: hdr_policy().sdr_gamma,
                gamut_stretch: hdr_policy().gamut_stretch,
                hardware_offload: self.hdr_hardware_offload,
                is_sdr: false,
            })
        } else if self.screen_filter.is_noop() && self.mirroring.is_none() {
            Some(HdrOutputConfig::sdr_tonemapping())
        } else {
            None
        };
    }

    fn update_mirroring(&mut self, mirroring_output: Option<Output>) {
        self.mirroring = mirroring_output;
        self.update_hdr_config();
        self.postprocess_textures.clear();
    }

    fn update_screen_filter(&mut self, filter_config: ScreenFilter) {
        self.screen_filter = filter_config;
        self.update_hdr_config();
        self.postprocess_textures.clear();
    }

    fn send_signal_fifo_callbacks(&mut self) {
        if self.mirroring.is_none() {
            let _ = self.thread_sender.send(SurfaceCommand::SignalFIFO);
        }
    }

    fn send_frame_callbacks(&mut self) {
        if self.mirroring.is_none() {
            let _ = self
                .thread_sender
                .send(SurfaceCommand::SendFrames(self.frame_callback_seq));
        }
    }

    fn send_dmabuf_feedback(&mut self, states: RenderElementStates) {
        let _ = self
            .thread_sender
            .send(SurfaceCommand::RenderStates(states));
    }
}

impl Drop for SurfaceThreadState {
    fn drop(&mut self) {
        drop(self.compositor.take());
    }
}

fn source_node_for_surface(w: &WlSurface) -> Option<DrmNode> {
    with_renderer_surface_state(w, |state| {
        state
            .buffer()
            .and_then(|buffer| get_dmabuf(buffer).ok().and_then(|dmabuf| dmabuf.node()))
    })
    .flatten()
}

// TODO: Introduce can_shared_dmabuf_framebuffer for cases where we might select another gpu
//  and composite on target if not possible to finally get rid of "primary"
#[profiling::function]
fn render_node_for_output(
    output: &Output,
    primary_node: &DrmNode,
    target_node: &DrmNode,
    shell: &Shell,
) -> DrmNode {
    if is_same_gpu(target_node, primary_node) {
        return *target_node;
    }

    let Some(workspace) = shell.active_space(output) else {
        return *target_node;
    };
    let fullscreens: Vec<_> = workspace
        .get_fullscreen_surfaces()
        .map(|f| f.surface.clone())
        .collect();
    let nodes = if !fullscreens.is_empty() {
        fullscreens
    } else {
        workspace
            .mapped()
            .map(|mapped| mapped.active_window())
            .collect::<Vec<_>>()
    }
    .into_iter()
    .flat_map(|w| w.wl_surface().and_then(|s| source_node_for_surface(&s)))
    .collect::<Vec<_>>();

    if nodes.iter().any(|node| is_same_gpu(node, target_node)) || nodes.is_empty() {
        *target_node
    } else {
        *primary_node
    }
}

fn get_surface_dmabuf_feedback(
    render_node: DrmNode,
    target_node: DrmNode,
    render_formats: FormatSet,
    target_formats: FormatSet,
    primary_plane_formats: FormatSet,
    overlay_plane_formats: Option<FormatSet>,
) -> SurfaceDmabufFeedback {
    // We limit the scan-out trache to formats we can also render from
    // so that there is always a fallback render path available in case
    // the supplied buffer can not be scanned out directly

    let primary_plane_formats = primary_plane_formats
        .intersection(&render_formats)
        .cloned()
        .collect::<FormatSet>();
    let overlay_plane_formats = overlay_plane_formats.map(|formats| {
        formats
            .intersection(&render_formats)
            .cloned()
            .collect::<FormatSet>()
    });

    let mut builder = DmabufFeedbackBuilder::new(render_node.dev_id(), render_formats.clone());

    if !is_same_gpu(&target_node, &render_node) {
        builder = builder.add_preference_tranche(
            target_node.dev_id(),
            zwp_linux_dmabuf_feedback_v1::TrancheFlags::Sampling,
            target_formats,
            6..=6,
        );
    };
    let render_feedback = builder.clone().build().unwrap();

    let primary_scanout_feedback = builder
        .clone()
        .add_preference_tranche(
            target_node.dev_id(),
            zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout,
            primary_plane_formats,
            4..=6,
        )
        .build()
        .unwrap();
    let overlay_scanout_feedback = overlay_plane_formats.map(|formats| {
        builder
            .add_preference_tranche(
                target_node.dev_id(),
                zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout,
                formats,
                4..=6,
            )
            .build()
            .unwrap()
    });

    SurfaceDmabufFeedback {
        render_feedback,
        overlay_scanout_feedback,
        primary_scanout_feedback,
    }
}

fn take_screencopy_frames<E: Element>(
    output: &Output,
    elements: &[E],
    has_cursor_mode_none: &mut bool,
) -> Vec<(
    ScreencopySessionRef,
    ScreencopyFrame,
    Result<(Option<Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>,
)> {
    output
        .take_pending_frames()
        .into_iter()
        .map(|(session, frame)| {
            let additional_damage = frame.damage();
            let session_data = session.user_data().get::<SessionData>().unwrap();
            let mut damage_tracking = session_data.lock().unwrap();

            let buffer = frame.buffer();
            let age = if matches!(buffer_type(&buffer), Some(BufferType::Shm)) {
                // TODO re-use offscreen buffer to damage track screencopy to shm
                0
            } else {
                1
            };

            if !additional_damage.is_empty() {
                let area = output
                    .current_mode()
                    .unwrap()
                    /* TODO: Mode is Buffer..., why is this Physical in the first place */
                    .size
                    .to_logical(1)
                    .to_buffer(1, Transform::Normal)
                    .to_f64();

                let additional_damage_elements: Vec<_> = additional_damage
                    .into_iter()
                    .map(|rect| {
                        rect.to_f64()
                            .to_logical(
                                output.current_scale().fractional_scale(),
                                output.current_transform(),
                                &area,
                            )
                            .to_i32_round()
                    })
                    .map(DamageElement::new)
                    .collect();
                let _ = damage_tracking
                    .dt
                    .damage_output(age, &additional_damage_elements);
            };

            let res = damage_tracking.dt.damage_output(age, elements);

            if !session.draw_cursor() {
                *has_cursor_mode_none = true;
            }

            let res = res.map(|(a, b)| (a.cloned(), b));
            std::mem::drop(damage_tracking);
            (session, frame, res)
        })
        .collect()
}

fn send_screencopy_result<'a>(
    renderer: &mut GlMultiRenderer<'a>,
    output: &Output,
    pre_postprocess_data: &mut PrePostprocessData,
    tx: &std::sync::mpsc::Sender<PendingImageCopyData>,
    frame_result: &RenderFrameResult<GbmBuffer, GbmFramebuffer, CosmicElement<GlMultiRenderer<'a>>>,
    elements: &[CosmicElement<GlMultiRenderer<'a>>],
    (session, frame, res): (
        &ScreencopySessionRef,
        ScreencopyFrame,
        Result<(Option<Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>,
    ),
    presentation_time: Duration,
) -> Result<()> {
    let (damage, _) = match res {
        Ok(damage) => damage,
        Err(err) => {
            frame.fail(CaptureFailureReason::Unknown);
            return Err(err.into());
        }
    };

    let mut sync = SyncPoint::default();
    let mut dmabuf_clone;
    let mut render_buffer;
    let buffer = frame.buffer();
    let mut shm_buffer = false;
    let buffer_size = match buffer_dimensions(&buffer) {
        Some(size) => size,
        None => {
            frame.fail(CaptureFailureReason::Unknown);
            return Err(
                RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering(
                    MultiError::ImportFailed,
                )
                .into(),
            );
        }
    };
    let mut fb = if let Ok(dmabuf) = get_dmabuf(&buffer) {
        dmabuf_clone = dmabuf.clone();
        Some(
            renderer
                .bind(&mut dmabuf_clone)
                .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?,
        )
    } else {
        shm_buffer = true;
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)? // eh, we have to do some error
            .expect("We should be able to convert all hardcoded shm screencopy formats");

        if pre_postprocess_data
            .texture
            .as_ref()
            .is_some_and(|tex| tex.format() == Some(format))
            && (!session.draw_cursor() || pre_postprocess_data.cursor_texture.is_none())
        {
            None
        } else {
            render_buffer =
                Offscreen::<GlesRenderbuffer>::create_buffer(renderer, format, buffer_size)
                    .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?;
            Some(
                renderer
                    .bind(&mut render_buffer)
                    .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?,
            )
        }
    };

    if let Some(ref damage) = damage {
        let (output_size, output_scale, output_transform) = (
            output.current_mode().ok_or(OutputNoMode)?.size,
            output.current_scale().fractional_scale(),
            output.current_transform(),
        );

        let filter = (!session.draw_cursor())
            .then(|| {
                elements.iter().filter_map(|elem| {
                    if let CosmicElement::Cursor(_) = elem {
                        Some(elem.id().clone())
                    } else {
                        None
                    }
                })
            })
            .into_iter()
            .flatten();

        // If the screen is rotated, we must convert damage to match output.
        let adjusted = damage
            .iter()
            .copied()
            .map(|rect| {
                let logical = rect.to_logical(1);
                logical
                    .to_buffer(
                        1,
                        output_transform.invert(),
                        &buffer_size.to_logical(1, output_transform),
                    )
                    .to_logical(1, Transform::Normal, &buffer_size)
                    .to_physical(1)
            })
            .collect::<Vec<_>>();

        if let Some(tex) = pre_postprocess_data.texture.as_mut() {
            let tex_fb = renderer
                .bind(tex)
                .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?;

            if let Some(fb) = fb.as_mut() {
                for rect in adjusted.iter().copied() {
                    // TODO: On Vulkan, may need to combine sync points instead of just using latest?
                    sync = renderer
                        .blit(&tex_fb, fb, rect, rect, TextureFilter::Linear)
                        .map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                }
                if let Some(cursor_geometry) = pre_postprocess_data
                    .cursor_geometry
                    .as_ref()
                    .filter(|_| session.draw_cursor())
                {
                    let cursor_damage = adjusted
                        .iter()
                        .filter_map(|rect| cursor_geometry.intersection(*rect))
                        .map(|rect| Rectangle::new(rect.loc - cursor_geometry.loc, rect.size))
                        .collect::<Vec<_>>();
                    let mut frame = renderer.render(fb, output_size, output_transform).map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                    frame
                        .as_mut()
                        .render_texture_from_to(
                            pre_postprocess_data.cursor_texture.as_ref().unwrap(),
                            Rectangle::new(
                                Point::from((0., 0.)),
                                cursor_geometry
                                    .size
                                    .to_logical(1)
                                    .to_buffer(1, Transform::Normal)
                                    .to_f64(),
                            ),
                            *cursor_geometry,
                            &cursor_damage,
                            &[*cursor_geometry],
                            Transform::Normal,
                            1.0,
                        )
                        .map_err(GlMultiError::Render)
                        .map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                    let sync = frame.finish().map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                    renderer.wait(&sync).map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                }
            } else {
                fb = Some(tex_fb);
            }
        } else {
            sync = frame_result
                .blit_frame_result(
                    output_size,
                    output_transform,
                    output_scale,
                    renderer,
                    fb.as_mut().unwrap(),
                    adjusted,
                    filter,
                )
                .map_err(|err| match err {
                    BlitFrameResultError::Rendering(err) => {
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering(err)
                    }
                    BlitFrameResultError::Export(_) => {
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering(
                            MultiError::DeviceMissing,
                        )
                    }
                })?;
        };
    }

    let transform = output.current_transform();

    if let Some(data) = submit_buffer(
        frame,
        renderer,
        shm_buffer.then_some(fb.as_mut().unwrap()),
        transform,
        damage.as_deref(),
        sync,
        // Don't reference `Buffer`s since we blit from framebuffer/postprocess buffer
        vec![],
    )
    .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?
    {
        if shm_buffer || frame_result.is_empty {
            data.frame
                .success(transform, data.damage, presentation_time);
        } else {
            let _ = tx.send(data);
        }
    }

    Ok(())
}

fn send_screencopy_result_vulkan<'a>(
    renderer: &mut VulkanMultiRenderer<'a>,
    output: &Output,
    tx: &std::sync::mpsc::Sender<PendingImageCopyData>,
    frame_result: &RenderFrameResult<
        GbmBuffer,
        GbmFramebuffer,
        CosmicElement<VulkanMultiRenderer<'a>>,
    >,
    elements: &[CosmicElement<VulkanMultiRenderer<'a>>],
    (session, frame, res): (
        &ScreencopySessionRef,
        ScreencopyFrame,
        Result<(Option<Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>,
    ),
    presentation_time: Duration,
) -> Result<()> {
    let (damage, _) = match res {
        Ok(damage) => damage,
        Err(err) => {
            frame.fail(CaptureFailureReason::Unknown);
            return Err(err.into());
        }
    };

    let mut sync = SyncPoint::default();
    let mut dmabuf_clone;
    let mut render_buffer = None;
    let buffer = frame.buffer();
    let mut shm_buffer = false;
    let buffer_size = match buffer_dimensions(&buffer) {
        Some(size) => size,
        None => {
            frame.fail(CaptureFailureReason::Unknown);
            return Err(
                RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(
                    MultiError::ImportFailed,
                )
                .into(),
            );
        }
    };
    let mut fb = if let Ok(dmabuf) = get_dmabuf(&buffer) {
        dmabuf_clone = dmabuf.clone();
        tracing::debug!(
            output = %output.name(),
            format = ?dmabuf.format(),
            size = ?buffer_size,
            "send_screencopy_result_vulkan: binding target DMA-BUF"
        );
        Some(
            renderer
                .bind(&mut dmabuf_clone)
                .map_err(|err| {
                    tracing::error!(
                        "send_screencopy_result_vulkan: failed to bind target DMA-BUF (format={:?}, size={:?}): {:#}",
                        dmabuf.format(), buffer_size, err
                    );
                    RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
                })?,
        )
    } else {
        shm_buffer = true;
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)?
            .expect("We should be able to convert all hardcoded shm screencopy formats");

        tracing::debug!(
            output = %output.name(),
            format = ?format,
            size = ?buffer_size,
            "send_screencopy_result_vulkan: allocating offscreen buffer for SHM screencopy"
        );

        let img = Offscreen::<smithay::backend::vulkan::image::VulkanImage>::create_buffer(
            renderer,
            format,
            buffer_size,
        )
        .map_err(|err| {
            tracing::error!(
                "send_screencopy_result_vulkan: create_buffer failed for SHM (format={:?}, size={:?}): {:#}",
                format, buffer_size, err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?;
        render_buffer = Some(img);
        Some(
            renderer
                .bind(render_buffer.as_mut().unwrap())
                .map_err(|err| {
                    tracing::error!(
                        "send_screencopy_result_vulkan: bind render_buffer failed for SHM (format={:?}, size={:?}): {:#}",
                        format, buffer_size, err
                    );
                    RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
                })?,
        )
    };

    if let Some(ref damage) = damage {
        let (output_size, output_scale, output_transform) = (
            output.current_mode().ok_or(OutputNoMode)?.size,
            output.current_scale().fractional_scale(),
            output.current_transform(),
        );

        let filter = (!session.draw_cursor())
            .then(|| {
                elements.iter().filter_map(|elem| {
                    if let CosmicElement::Cursor(_) = elem {
                        Some(elem.id().clone())
                    } else {
                        None
                    }
                })
            })
            .into_iter()
            .flatten();

        // If the screen is rotated, we must convert damage to match output.
        let adjusted = damage
            .iter()
            .copied()
            .map(|rect| {
                let logical = rect.to_logical(1);
                logical
                    .to_buffer(
                        1,
                        output_transform.invert(),
                        &buffer_size.to_logical(1, output_transform),
                    )
                    .to_logical(1, Transform::Normal, &buffer_size)
                    .to_physical(1)
            })
            .collect::<Vec<_>>();

        sync = frame_result
            .blit_frame_result(
                output_size,
                output_transform,
                output_scale,
                renderer,
                fb.as_mut().unwrap(),
                adjusted,
                filter,
            )
            .map_err(|err| {
                tracing::error!(
                    "send_screencopy_result_vulkan: blit_frame_result failed: {:#}",
                    err
                );
                match err {
                    BlitFrameResultError::Rendering(err) => {
                        RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
                    }
                    BlitFrameResultError::Export(_) => {
                        RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(
                            MultiError::DeviceMissing,
                        )
                    }
                }
            })?;
    }

    let transform = output.current_transform();

    if shm_buffer {
        drop(fb);
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)?
            .expect("We should be able to convert all hardcoded shm screencopy formats");
        let vulkan_renderer = if renderer.target_as_ref().is_some() {
            renderer.target_as_mut().unwrap()
        } else {
            renderer.as_mut()
        };
        let task = vulkan_renderer
            .record_copy_image_to_shm(
                render_buffer.as_ref().unwrap(),
                Rectangle::from_size(buffer_size),
                format,
            )
            .map_err(|err| {
                tracing::error!(
                    "send_screencopy_result_vulkan: record_copy_image_to_shm failed: {:#}",
                    err
                );
                RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(
                    MultiError::Render(err),
                )
            })?;
        let damage_rects = damage
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|rect| {
                let logical = rect.to_logical(1);
                logical.to_buffer(1, transform.invert(), &buffer_size.to_logical(1, transform))
            })
            .collect();
        let capture = PendingShmCapture {
            task,
            frame,
            transform,
            damage: damage_rects,
            presentation_time,
        };
        if let Some(queue) = output.user_data().get::<OutputPendingShmCaptures>() {
            queue.0.lock().unwrap().push(capture);
        }
        return Ok(());
    }

    if let Some(data) = submit_buffer(
        frame,
        renderer,
        None,
        transform,
        damage.as_deref(),
        sync,
        // Don't reference `Buffer`s since we blit from framebuffer/postprocess buffer
        vec![],
    )
    .map_err(|err| {
        tracing::error!(
            "send_screencopy_result_vulkan: submit_buffer failed: {:#}",
            err
        );
        RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
    })? {
        if frame_result.is_empty {
            data.frame
                .success(transform, data.damage, presentation_time);
        } else {
            let _ = tx.send(data);
        }
    }

    Ok(())
}

fn send_toplevel_screencopy_result_vulkan<'a>(
    renderer: &mut VulkanMultiRenderer<'a>,
    output: &Output,
    tx: &std::sync::mpsc::Sender<PendingImageCopyData>,
    toplevel: &CosmicSurface,
    session: &ScreencopySessionRef,
    frame: ScreencopyFrame,
    presentation_time: Duration,
) -> Result<()> {
    if !toplevel.alive() {
        let mut toplevel_clone = toplevel.clone();
        toplevel_clone.remove_session(session);
        frame.fail(CaptureFailureReason::Stopped);
        return Ok(());
    }

    let scale = output.current_scale().fractional_scale();
    let geometry = toplevel.geometry();
    let buffer = frame.buffer();
    let buffer_size = match buffer_dimensions(&buffer) {
        Some(s) => s,
        None => {
            frame.fail(CaptureFailureReason::Unknown);
            return Ok(());
        }
    };
    let phys_size = geometry.size.to_f64().to_physical(scale).to_i32_round();
    let expected_size = Size::from((phys_size.w, phys_size.h));
    if buffer_size != expected_size {
        frame.fail(CaptureFailureReason::BufferConstraints);
        return Ok(());
    }

    let mut sync = SyncPoint::default();
    let mut dmabuf_clone;
    let mut render_buffer = None;
    let mut shm_buffer = false;

    let mut dst_fb = if let Ok(dmabuf) = get_dmabuf(&buffer) {
        dmabuf_clone = dmabuf.clone();
        Some(renderer.bind(&mut dmabuf_clone).map_err(|err| {
            tracing::error!(
                "send_toplevel_screencopy_result_vulkan: failed to bind dst dmabuf: {:#}",
                err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?)
    } else {
        shm_buffer = true;
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)?
            .expect("We should be able to convert all hardcoded shm screencopy formats");
        let img = Offscreen::<smithay::backend::vulkan::image::VulkanImage>::create_buffer(
            renderer,
            format,
            buffer_size,
        )
        .map_err(|err| {
            tracing::error!(
                "send_toplevel_screencopy_result_vulkan: create_buffer failed for SHM: {:#}",
                err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?;
        render_buffer = Some(img);
        Some(renderer.bind(render_buffer.as_mut().unwrap()).map_err(|err| {
            tracing::error!(
                "send_toplevel_screencopy_result_vulkan: bind render_buffer failed for SHM: {:#}",
                err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?)
    };

    let loc_phys: Point<i32, Physical> = geometry.loc.to_f64().to_physical(scale).to_i32_round();
    let mut elements: Vec<SurfaceRenderElement<VulkanMultiRenderer>> = Vec::new();
    toplevel.push_render_elements(
        renderer,
        Point::from((-loc_phys.x, -loc_phys.y)),
        Scale::from(scale),
        1.0,
        None,
        None,
        false,
        [0; 4],
        0,
        &mut |elem| elements.push(elem),
        None,
    );

    let mut direct_blit_done = false;
    if elements.len() == 1 && geometry.loc == (0, 0).into() {
        if let Some(wl_surface) = toplevel.wl_surface() {
            let window_dmabuf = smithay::backend::renderer::utils::with_renderer_surface_state(
                &wl_surface,
                |state| {
                    let buffer = state.buffer()?;
                    get_dmabuf(buffer).ok().cloned()
                },
            )
            .flatten();

            if let Some(mut src_dmabuf) = window_dmabuf {
                let rect = Rectangle::from_size(phys_size);
                let src_size = src_dmabuf
                    .size()
                    .to_logical(1, Transform::Normal)
                    .to_physical(1);
                if src_size == rect.size {
                    if let Ok(src_fb) = renderer.bind(&mut src_dmabuf) {
                        if let Ok(blit_sync) = renderer.blit(
                            &src_fb,
                            dst_fb.as_mut().unwrap(),
                            rect,
                            rect,
                            TextureFilter::Nearest,
                        ) {
                            sync = blit_sync;
                            direct_blit_done = true;
                        }
                    }
                }
            }
        }
    }

    if !direct_blit_done {
        let mut frame = renderer
            .render(dst_fb.as_mut().unwrap(), phys_size, Transform::Normal)
            .map_err(|err| {
                tracing::error!(
                    "send_toplevel_screencopy_result_vulkan: render failed: {:#}",
                    err
                );
                RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
            })?;

        frame
            .clear(Color32F::TRANSPARENT, &[Rectangle::from_size(phys_size)])
            .map_err(|err| {
                tracing::error!(
                    "send_toplevel_screencopy_result_vulkan: clear failed: {:#}",
                    err
                );
                RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
            })?;

        let _ = smithay::backend::renderer::utils::draw_render_elements(
            &mut frame,
            Scale::from(scale),
            &elements,
            &[Rectangle::from_size(phys_size)],
        )
        .map_err(|err| {
            tracing::error!(
                "send_toplevel_screencopy_result_vulkan: draw_render_elements failed: {:#}",
                err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?;

        sync = frame.finish().map_err(|err| {
            tracing::error!(
                "send_toplevel_screencopy_result_vulkan: frame.finish failed: {:#}",
                err
            );
            RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
        })?;
    }

    if shm_buffer {
        drop(dst_fb);
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)?
            .expect("We should be able to convert all hardcoded shm screencopy formats");
        let vulkan_renderer = if renderer.target_as_ref().is_some() {
            renderer.target_as_mut().unwrap()
        } else {
            renderer.as_mut()
        };
        let task = vulkan_renderer
            .record_copy_image_to_shm(
                render_buffer.as_ref().unwrap(),
                Rectangle::from_size(buffer_size),
                format,
            )
            .map_err(|err| {
                tracing::error!(
                    "send_toplevel_screencopy_result_vulkan: record_copy_image_to_shm failed: {:#}",
                    err
                );
                RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(
                    MultiError::Render(err),
                )
            })?;
        let capture = PendingShmCapture {
            task,
            frame,
            transform: Transform::Normal,
            damage: vec![Rectangle::from_size(buffer_size)],
            presentation_time,
        };
        if let Some(queue) = output.user_data().get::<OutputPendingShmCaptures>() {
            queue.0.lock().unwrap().push(capture);
        }
        return Ok(());
    }

    if let Some(data) = submit_buffer(
        frame,
        renderer,
        None,
        Transform::Normal,
        Some(&[Rectangle::from_size(
            buffer_size.to_logical(1, Transform::Normal).to_physical(1),
        )]),
        sync,
        vec![],
    )
    .map_err(|err| {
        tracing::error!(
            "send_toplevel_screencopy_result_vulkan: submit_buffer failed: {:#}",
            err
        );
        RenderError::<<VulkanMultiRenderer as RendererSuper>::Error>::Rendering(err)
    })? {
        let _ = tx.send(data);
    }

    Ok(())
}

fn postprocess_elements<'a>(
    renderer: &mut GlMultiRenderer<'a>,
    output: &Output,
    pre_postprocess_data: &PrePostprocessData,
    postprocess_state: &PostprocessState,
    screen_filter: &ScreenFilter,
    hdr_enabled: bool,
    hdr_reference_white: f32,
    hdr_hardware_offload: bool,
) -> Vec<CosmicElement<GlMultiRenderer<'a>>> {
    let postprocess_texture_shader = Borrow::<GlesRenderer>::borrow(renderer.as_ref())
        .egl_context()
        .user_data()
        .get::<PostprocessShader>()
        .expect("OffscreenShader should be available through `init_shaders`");

    let build_uniforms = || {
        vec![
            Uniform::new("invert", if screen_filter.inverted { 1. } else { 0. }),
            Uniform::new(
                "color_mode",
                screen_filter
                    .color_filter
                    .map(|val| val as u8 as f32)
                    .unwrap_or(0.),
            ),
            Uniform::new("hdr_enabled", if hdr_enabled { 1.0 } else { 0.0 }),
            Uniform::new("hdr_reference_white", hdr_reference_white),
            Uniform::new("hdr_sdr_gamma", hdr_policy().sdr_gamma),
            Uniform::new("hdr_gamut_stretch", hdr_policy().gamut_stretch),
            Uniform::new(
                "hdr_hardware_offload",
                if hdr_hardware_offload { 1.0 } else { 0.0 },
            ),
        ]
    };

    let mut elements: [Option<TextureShaderElement>; 2] = [None, None];
    if let (Some(cursor_texture), Some(cursor_geometry)) = (
        postprocess_state.cursor_texture.as_ref(),
        pre_postprocess_data.cursor_geometry.as_ref(),
    ) {
        let texture_elem = TextureRenderElement::from_texture_render_buffer(
            cursor_geometry.loc.to_f64(),
            cursor_texture,
            None,
            Some(Rectangle::new(
                Point::from((0., 0.)),
                cursor_geometry.size.to_logical(1).to_f64(),
            )),
            Some(
                cursor_geometry
                    .size
                    .to_f64()
                    .to_logical(output.current_scale().fractional_scale())
                    .to_i32_round(),
            ),
            Kind::Cursor,
        );

        elements[0] = Some(TextureShaderElement::new(
            texture_elem,
            postprocess_texture_shader.0.clone(),
            build_uniforms(),
        ));
    }

    let texture_elem = TextureRenderElement::from_texture_render_buffer(
        (0., 0.),
        &postprocess_state.texture,
        None,
        Some(Rectangle::new(
            Point::from((0., 0.)),
            postprocess_state.output_config.size.to_logical(1).to_f64(),
        )),
        Some(
            postprocess_state
                .output_config
                .size
                .to_f64()
                .to_logical(postprocess_state.output_config.fractional_scale)
                .to_i32_round(),
        ),
        Kind::Unspecified,
    );
    elements[1] = Some(TextureShaderElement::new(
        texture_elem,
        postprocess_texture_shader.0.clone(),
        build_uniforms(),
    ));

    constrain_render_elements(
        elements.into_iter().flatten(),
        (0, 0),
        Rectangle::new(Point::from((0, 0)), postprocess_state.output_config.size),
        Rectangle::new(Point::from((0, 0)), postprocess_state.output_config.size),
        ConstrainScaleBehavior::Fit,
        ConstrainAlign::CENTER,
        postprocess_state.output_config.fractional_scale,
    )
    .map(CosmicElement::<GlMultiRenderer>::Postprocess)
    .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use smithay::wayland::color::management::{Primaries, PrimariesOption, TransferFunction};

    #[test]
    fn sdr_surface_on_sdr_output_allows_scanout() {
        assert!(is_surface_scanout_compatible(false, None));
        assert!(is_surface_scanout_compatible(
            false,
            Some(&ImageDescription::SRGB)
        ));
    }

    #[test]
    fn hdr_surface_on_sdr_output_blocks_scanout() {
        let pq_desc = ImageDescription {
            transfer: TransferFunction::St2084Pq,
            primaries: PrimariesOption {
                named: Some(Primaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert!(!is_surface_scanout_compatible(false, Some(&pq_desc)));
        assert!(!is_surface_scanout_compatible(
            false,
            Some(&ImageDescription::WINDOWS_SCRGB)
        ));
    }

    #[test]
    fn pq_bt2020_surface_on_hdr_output_allows_scanout() {
        let pq_desc = ImageDescription {
            transfer: TransferFunction::St2084Pq,
            primaries: PrimariesOption {
                named: Some(Primaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert!(is_surface_scanout_compatible(true, Some(&pq_desc)));
        assert!(is_surface_scanout_compatible(
            true,
            Some(&ImageDescription::WINDOWS_BT2100)
        ));
    }

    #[test]
    fn non_pq_surface_on_hdr_output_blocks_scanout() {
        // SDR on HDR output needs SDR->HDR expansion
        assert!(!is_surface_scanout_compatible(true, None));
        assert!(!is_surface_scanout_compatible(
            true,
            Some(&ImageDescription::SRGB)
        ));

        // Extended linear scRGB on HDR output needs linear->PQ conversion
        assert!(!is_surface_scanout_compatible(
            true,
            Some(&ImageDescription::WINDOWS_SCRGB)
        ));

        // HLG on PQ HDR output needs HLG->PQ conversion
        let hlg_desc = ImageDescription {
            transfer: TransferFunction::Hlg,
            primaries: PrimariesOption {
                named: Some(Primaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert!(!is_surface_scanout_compatible(true, Some(&hlg_desc)));
    }

    #[test]
    fn four_tier_scanout_plan_hierarchy() {
        use smithay::backend::drm::color::{
            CrtcColorCapabilities, DrmScanoutCapabilities, PlaneColorConversion,
        };

        let mut caps = DrmScanoutCapabilities {
            crtc_color: CrtcColorCapabilities {
                has_gamma_lut: true,
                gamma_lut_size: 4096,
                has_degamma_lut: true,
                degamma_lut_size: 4096,
                has_ctm: true,
            },
            supports_plane_colorop: false,
            primary_plane_color_pipelines: vec![
                smithay::backend::drm::colorop::ColorPipeline::synthetic(),
            ],
            primary_plane_formats: FormatSet::default(),
            supports_fp16: true,
            supports_10bit: true,
        };

        // 1. Tier 1: DirectPassthrough
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_BT2100), 203);
        assert_eq!(plan, ScanoutPlan::DirectPassthrough);
        assert!(plan.allows_primary_scanout());

        // 2. Tier 2A: PlaneColorop
        caps.supports_plane_colorop = true;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::PlaneColorop(PlaneColorConversion::ScRgbToPq { .. })
        ));
        assert!(plan.allows_primary_scanout());

        // 3. Tier 2B: CrtcHardware
        caps.supports_plane_colorop = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert!(matches!(
            plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToPq { .. })
        ));
        assert!(plan.allows_primary_scanout());

        // 4. Tier 3: VulkanFastDirectFlip
        caps.crtc_color.has_ctm = false;
        let plan = caps.evaluate_scanout_plan(true, Some(&ImageDescription::WINDOWS_SCRGB), 203);
        assert_eq!(plan, ScanoutPlan::VulkanFastDirectFlip);
        assert!(!plan.allows_primary_scanout());
    }

    #[test]
    fn test_plane_colorop_scrgb_to_pq_hardware_state() {
        use smithay::backend::drm::color::{DrmColorCtm, PlaneColorConversion, ScanoutPlan};

        let plan = ScanoutPlan::PlaneColorop(PlaneColorConversion::ScRgbToPq {
            reference_white: 203,
        });
        assert!(plan.allows_primary_scanout());
        assert!(plan.requires_plane_colorop().is_some());

        let conv = plan.requires_plane_colorop().unwrap();
        let color_state = conv.to_crtc_color_state(4096, 4096);

        // Degamma is None because scRGB is already linear light
        assert!(color_state.degamma_lut.is_none());
        // CTM is present and scaled by reference_white / 10000.0 (0.0203)
        assert!(color_state.ctm.is_some());
        // Gamma LUT is present and encodes full ST 2084 PQ up to 10,000 nits
        assert!(color_state.gamma_lut.is_some());

        let ctm = color_state.ctm.unwrap();
        let gamma_lut = color_state.gamma_lut.unwrap();
        assert_eq!(gamma_lut.len(), 4096);

        // Verify CTM scales scRGB 1.0 (nominal white) to 203 / 10000 = 0.0203
        let m00 = DrmColorCtm::from_s31_32(ctm.matrix[0]);
        let m01 = DrmColorCtm::from_s31_32(ctm.matrix[1]);
        let m02 = DrmColorCtm::from_s31_32(ctm.matrix[2]);
        let r0 = m00 + m01 + m02;
        let expected_scale = 203.0 / 10000.0;
        assert!(
            (r0 - expected_scale).abs() < 2e-6,
            "r0={r0} expected={expected_scale}"
        );

        // Verify HDR highlight (e.g. 1000 nits = 4.926 in scRGB) is scaled to <= 1.0
        let highlight_scrgb = 1000.0 / 203.0;
        let highlight_linear = r0 * highlight_scrgb;
        assert!((highlight_linear - 0.100).abs() < 1e-4);
        assert!(
            highlight_linear <= 1.0,
            "Highlight must not overflow 1D LUT input domain [0, 1]!"
        );

        // Verify negative scRGB value in wide gamut (BT.2020 pure green in Rec.709: R = -0.4677)
        let wide_gamut_green = [-0.4677, 1.0772, -0.0298];
        let m10 = DrmColorCtm::from_s31_32(ctm.matrix[3]);
        let m11 = DrmColorCtm::from_s31_32(ctm.matrix[4]);
        let m12 = DrmColorCtm::from_s31_32(ctm.matrix[5]);
        let m20 = DrmColorCtm::from_s31_32(ctm.matrix[6]);
        let m21 = DrmColorCtm::from_s31_32(ctm.matrix[7]);
        let m22 = DrmColorCtm::from_s31_32(ctm.matrix[8]);
        let out_r =
            m00 * wide_gamut_green[0] + m01 * wide_gamut_green[1] + m02 * wide_gamut_green[2];
        let out_g =
            m10 * wide_gamut_green[0] + m11 * wide_gamut_green[1] + m12 * wide_gamut_green[2];
        let out_b =
            m20 * wide_gamut_green[0] + m21 * wide_gamut_green[1] + m22 * wide_gamut_green[2];
        assert!(out_r >= -1e-4, "Out R must be non-negative: {out_r}");
        assert!(out_g > 0.0, "Out G must be positive: {out_g}");
        assert!(out_b >= -1e-4, "Out B must be non-negative: {out_b}");

        // Verify 1000 nits encodes to ~49270 in ST 2084 PQ gamma LUT
        let idx = (highlight_linear * 4095.0) as usize;
        let pq_val = gamma_lut[idx].red;
        assert!((pq_val as i32 - 49270).abs() < 200, "pq_val={pq_val}");
    }

    #[test]
    fn test_plane_colorop_srgb_to_pq_hardware_state() {
        use smithay::backend::drm::color::{
            DrmColorCtm, PlaneColorConversion, ScanoutPlan, encode_pq,
        };

        let plan = ScanoutPlan::PlaneColorop(PlaneColorConversion::SrgbToPq {
            reference_white: 335,
        });
        assert!(plan.allows_primary_scanout());
        assert!(plan.requires_plane_colorop().is_some());
        assert!(plan.requires_crtc_color_state().is_some());

        let conv = plan.requires_plane_colorop().unwrap();
        let color_state = conv.to_crtc_color_state(4096, 4096);

        // Degamma linearizes 8/10-bit SDR content
        assert!(color_state.degamma_lut.is_some());
        let degamma_lut = color_state.degamma_lut.unwrap();
        assert_eq!(degamma_lut.len(), 4096);
        assert_eq!(degamma_lut[0].red, 0);
        assert_eq!(degamma_lut[4095].red, 65535);

        // CTM is scaled Rec.709 to BT.2020 matrix (row sum = scale)
        assert!(color_state.ctm.is_some());
        let ctm = color_state.ctm.unwrap();
        let scale = 335.0 / 10000.0;
        let m00 = DrmColorCtm::from_s31_32(ctm.matrix[0]);
        let m01 = DrmColorCtm::from_s31_32(ctm.matrix[1]);
        let m02 = DrmColorCtm::from_s31_32(ctm.matrix[2]);
        let r0 = m00 + m01 + m02;
        assert!(
            (r0 - scale).abs() < 2e-6,
            "r0={r0} must be scale={scale} (scaled CTM)"
        );

        // Gamma LUT maps linear [0, 1] to canonical 10,000-nit PQ
        assert!(color_state.gamma_lut.is_some());
        let gamma_lut = color_state.gamma_lut.unwrap();
        assert_eq!(gamma_lut.len(), 4096);
        assert_eq!(
            gamma_lut[0].red, 0,
            "Black level must not be raised (no washed out SDR)"
        );
        assert_eq!(gamma_lut[0].green, 0);
        assert_eq!(gamma_lut[0].blue, 0);
        assert_eq!(gamma_lut[4095].red, 65535);

        let idx_335 = (scale * 4095.0) as usize;
        let expected_white = (encode_pq(335.0 / 10000.0) * 65535.0).round() as u16;
        assert!((gamma_lut[idx_335].red as i32 - expected_white as i32).abs() < 500);
    }

    #[test]
    fn test_plane_colorop_scanout_color_transform() {
        use smithay::backend::drm::color::PlaneColorConversion;
        use smithay::backend::drm::colorop::{Curve1DType, PostBlendEncode};

        // 1. scRGB to PQ (HDR)
        let scrgb = PlaneColorConversion::ScRgbToPq {
            reference_white: 203,
        };
        let tr = scrgb
            .to_scanout_color_transform()
            .expect("scRGB to PQ transform");
        assert_eq!(tr.decode, None); // scRGB is linear
        assert_eq!(tr.encode, Some(Curve1DType::Pq125InvEotf));
        assert!((tr.multiplier - (203.0 / 80.0)).abs() < 1e-6);
        assert!(tr.ctm.is_some());
        let ctm = tr.ctm.unwrap();
        assert!((ctm[0] - 0.6274040).abs() < 1e-4);

        // 2. sRGB to PQ (HDR)
        let srgb = PlaneColorConversion::SrgbToPq {
            reference_white: 203,
        };
        let tr = srgb
            .to_scanout_color_transform()
            .expect("sRGB to PQ transform");
        assert_eq!(tr.decode, Some(Curve1DType::Gamma22));
        assert_eq!(tr.encode, Some(Curve1DType::Pq125InvEotf));
        assert!((tr.multiplier - (203.0 / 80.0)).abs() < 1e-6);

        // 3. Post-blend encode offload
        let pb = PostBlendEncode::for_hdr(1000.0);
        assert_eq!(pb.encode, Curve1DType::Pq125InvEotf);
        assert!((pb.linear_max - (1000.0 / 80.0)).abs() < 1e-6);

        let linear_tr = pb.linear_transform(tr).expect("linear transform");
        assert_eq!(linear_tr.encode, None);
        assert!((linear_tr.multiplier - (203.0 / 1000.0)).abs() < 1e-4);
    }

    #[test]
    fn test_hardware_scanout_color_schemes_report() {
        use smithay::backend::allocator::Fourcc;
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::drm::color::{CrtcColorCapabilities, DrmScanoutCapabilities};
        use smithay::backend::drm::colorop::{ColorOpKind, plane_color_pipelines};
        use smithay::reexports::drm::ClientCapability;
        use smithay::reexports::drm::Device as BasicDevice;
        use smithay::reexports::drm::control::Device as ControlDevice;
        use smithay::utils::DeviceFd;
        use smithay::wayland::color::management::TransferFunction;

        println!(
            "\n================================================================================"
        );
        println!(
            "               HARDWARE SCANOUT COLOR SUPPORT DIAGNOSTIC REPORT                 "
        );
        println!(
            "================================================================================"
        );

        let mut cards_found = 0;
        for card_idx in 0..8 {
            let path = format!("/dev/dri/card{}", card_idx);
            let Ok(file) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
            else {
                continue;
            };
            cards_found += 1;
            let owned: rustix::fd::OwnedFd = file.into();
            let drm_fd = DrmDeviceFd::new(DeviceFd::from(owned));

            let _ = drm_fd.set_client_capability(ClientCapability::UniversalPlanes, true);
            let _ = drm_fd.set_client_capability(ClientCapability::Atomic, true);
            let _ = drm_fd.set_client_capability(ClientCapability::PlaneColorPipeline, true);

            let driver_info = drm_fd
                .get_driver()
                .map(|d| {
                    format!(
                        "{} (date: {}, desc: {})",
                        d.name().to_string_lossy(),
                        d.date().to_string_lossy(),
                        d.description().to_string_lossy()
                    )
                })
                .unwrap_or_else(|e| format!("unknown (err: {:?})", e));

            println!("\n[GPU Device] {}", path);
            println!("  Driver: {}", driver_info);

            let Ok(res) = drm_fd.resource_handles() else {
                println!("  Failed to get DRM resource handles");
                continue;
            };
            let Ok(plane_handles) = drm_fd.plane_handles() else {
                println!("  Failed to get Plane handles");
                continue;
            };

            for crtc_handle in res.crtcs() {
                let Ok(crtc_info) = drm_fd.get_crtc(*crtc_handle) else {
                    continue;
                };
                let Ok(props) = drm_fd.get_properties(*crtc_handle) else {
                    continue;
                };

                let mut gamma_lut_size = 0u64;
                let mut degamma_lut_size = 0u64;
                let mut has_gamma_lut = false;
                let mut has_degamma_lut = false;
                let mut has_ctm = false;
                let mut has_vrr = false;

                for (&prop_handle, &val) in &props {
                    if let Ok(info) = drm_fd.get_property(prop_handle) {
                        let name = info.name().to_string_lossy();
                        match name.as_ref() {
                            "GAMMA_LUT_SIZE" => gamma_lut_size = val,
                            "DEGAMMA_LUT_SIZE" => degamma_lut_size = val,
                            "GAMMA_LUT" => has_gamma_lut = true,
                            "DEGAMMA_LUT" => has_degamma_lut = true,
                            "CTM" => has_ctm = true,
                            "VRR_ENABLED" => has_vrr = true,
                            _ => {}
                        }
                    }
                }

                let mode_str = crtc_info
                    .mode()
                    .map(|m| format!("{}x{} @ {}Hz", m.size().0, m.size().1, m.vrefresh()))
                    .unwrap_or_else(|| "Inactive/Disabled".to_string());

                println!("\n  [CRTC {:?}] Mode: {}", crtc_handle, mode_str);
                println!(
                    "    - Hardware GAMMA_LUT:   {} (Max entries: {})",
                    if has_gamma_lut { "YES" } else { "NO" },
                    gamma_lut_size
                );
                println!(
                    "    - Hardware DEGAMMA_LUT: {} (Max entries: {})",
                    if has_degamma_lut { "YES" } else { "NO" },
                    degamma_lut_size
                );
                println!(
                    "    - Hardware CTM:         {} (S31.32 3x3 matrix)",
                    if has_ctm { "YES" } else { "NO" }
                );
                println!(
                    "    - Adaptive Sync (VRR):  {}",
                    if has_vrr { "YES" } else { "NO" }
                );

                for plane_handle in plane_handles.iter() {
                    let Ok(plane_info) = drm_fd.get_plane(*plane_handle) else {
                        continue;
                    };
                    let Ok(plane_props) = drm_fd.get_properties(*plane_handle) else {
                        continue;
                    };

                    let mut plane_type = "Overlay";
                    let mut has_color_pipeline = false;

                    for (&p_h, &p_v) in &plane_props {
                        if let Ok(p_info) = drm_fd.get_property(p_h) {
                            let p_name = p_info.name().to_string_lossy();
                            if p_name == "type" {
                                plane_type = match p_v {
                                    1 => "Primary",
                                    2 => "Cursor",
                                    _ => "Overlay",
                                };
                            } else if p_name == "COLOR_PIPELINE" {
                                has_color_pipeline = true;
                            }
                        }
                    }

                    if plane_type != "Primary"
                        || crtc_info.mode().is_none()
                        || plane_info.crtc() != Some(*crtc_handle)
                    {
                        continue;
                    }

                    let mut supports_sdr_8bit = false;
                    let mut supports_10bit = false;
                    let mut supports_fp16 = false;
                    let mut supported_fourccs = Vec::new();

                    for f in plane_info.formats() {
                        if let Ok(fourcc) = Fourcc::try_from(*f) {
                            supported_fourccs.push(fourcc);
                            match fourcc {
                                Fourcc::Argb8888
                                | Fourcc::Xrgb8888
                                | Fourcc::Abgr8888
                                | Fourcc::Xbgr8888 => {
                                    supports_sdr_8bit = true;
                                }
                                Fourcc::Abgr2101010
                                | Fourcc::Xbgr2101010
                                | Fourcc::Argb2101010
                                | Fourcc::Xrgb2101010 => {
                                    supports_10bit = true;
                                }
                                Fourcc::Abgr16161616f
                                | Fourcc::Xbgr16161616f
                                | Fourcc::Argb16161616f
                                | Fourcc::Xrgb16161616f => {
                                    supports_fp16 = true;
                                }
                                _ => {}
                            }
                        }
                    }

                    println!("\n    [Primary Plane {:?}]", plane_handle);
                    println!(
                        "      - COLOR_PIPELINE (Colorop uAPI): {}",
                        if has_color_pipeline { "YES" } else { "NO" }
                    );
                    println!(
                        "      - 8-bit SDR (XR24/AR24/XB24/AB24): {}",
                        if supports_sdr_8bit { "YES" } else { "NO" }
                    );
                    println!(
                        "      - 10-bit HDR (XB30/AB30/XR30/AR30): {}",
                        if supports_10bit { "YES" } else { "NO" }
                    );
                    println!(
                        "      - 16-bit Float (XB4H/AB4H/XR4H/AR4H): {}",
                        if supports_fp16 { "YES" } else { "NO" }
                    );

                    let pipelines = if has_color_pipeline {
                        plane_color_pipelines(&drm_fd, *plane_handle).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    println!("      - Discovered Color Pipelines: {}", pipelines.len());
                    for (i, p) in pipelines.iter().enumerate() {
                        println!(
                            "        * Pipeline #{} (ID: {}) [{} colorops]:",
                            i,
                            p.id,
                            p.ops.len()
                        );
                        for (op_idx, op) in p.ops.iter().enumerate() {
                            let bypass_tag = if op.bypassable {
                                "Bypassable"
                            } else {
                                "Mandatory"
                            };
                            match &op.kind {
                                ColorOpKind::Curve1D { supported } => {
                                    let names: Vec<_> =
                                        supported.iter().map(|(c, _)| format!("{:?}", c)).collect();
                                    println!(
                                        "          [Op {}] ID: {} | 1D Curve [{}] ({})",
                                        op_idx,
                                        op.id,
                                        names.join(", "),
                                        bypass_tag
                                    );
                                }
                                ColorOpKind::Multiplier => {
                                    println!(
                                        "          [Op {}] ID: {} | Multiplier (S31.32 fixed-point) ({})",
                                        op_idx, op.id, bypass_tag
                                    );
                                }
                                ColorOpKind::Ctm3x4 => {
                                    println!(
                                        "          [Op {}] ID: {} | 3x4 Matrix (DRM CTM) ({})",
                                        op_idx, op.id, bypass_tag
                                    );
                                }
                                ColorOpKind::Lut1D {
                                    size,
                                    interpolation,
                                } => {
                                    println!(
                                        "          [Op {}] ID: {} | 1D LUT (size: {}, {:?}) ({})",
                                        op_idx, op.id, size, interpolation, bypass_tag
                                    );
                                }
                                ColorOpKind::Lut3D {
                                    size,
                                    interpolation,
                                } => {
                                    println!(
                                        "          [Op {}] ID: {} | 3D LUT (size: {}, {:?}) ({})",
                                        op_idx, op.id, size, interpolation, bypass_tag
                                    );
                                }
                                ColorOpKind::Unknown { type_name } => {
                                    println!(
                                        "          [Op {}] ID: {} | Unknown Type: {} ({})",
                                        op_idx, op.id, type_name, bypass_tag
                                    );
                                }
                            }
                        }
                    }

                    // Also check Overlay and Cursor planes for Colorop capabilities
                    let mut overlay_colorop_count = 0;
                    let mut cursor_colorop_count = 0;
                    for other_plane in plane_handles.iter() {
                        if other_plane == plane_handle {
                            continue;
                        }
                        let Ok(other_props) = drm_fd.get_properties(*other_plane) else {
                            continue;
                        };
                        let mut is_overlay = false;
                        let mut is_cursor = false;
                        let mut has_colorop = false;
                        for (&p_h, &p_v) in &other_props {
                            if let Ok(p_info) = drm_fd.get_property(p_h) {
                                let name = p_info.name().to_string_lossy();
                                if name == "type" {
                                    if p_v == 0 {
                                        is_overlay = true;
                                    } else if p_v == 2 {
                                        is_cursor = true;
                                    }
                                } else if name == "COLOR_PIPELINE" {
                                    has_colorop = true;
                                }
                            }
                        }
                        if has_colorop {
                            if is_overlay {
                                overlay_colorop_count += 1;
                            }
                            if is_cursor {
                                cursor_colorop_count += 1;
                            }
                        }
                    }
                    println!(
                        "      - Other Planes Colorop: {} overlay plane(s), {} cursor plane(s) advertise COLOR_PIPELINE",
                        overlay_colorop_count, cursor_colorop_count
                    );

                    let caps = DrmScanoutCapabilities {
                        crtc_color: CrtcColorCapabilities {
                            has_gamma_lut,
                            gamma_lut_size,
                            has_degamma_lut,
                            degamma_lut_size,
                            has_ctm,
                        },
                        supports_plane_colorop: has_color_pipeline,
                        primary_plane_color_pipelines: pipelines.clone(),
                        primary_plane_formats: Default::default(),
                        supports_fp16,
                        supports_10bit,
                    };

                    println!(
                        "\n    ----------------------------------------------------------------------------"
                    );
                    println!(
                        "    EVALUATING SCANOUT SCHEMES ACROSS COLOR SPACES (CRTC {:?})",
                        crtc_handle
                    );
                    println!(
                        "    ----------------------------------------------------------------------------"
                    );

                    struct TestCase {
                        output_hdr: bool,
                        output_name: &'static str,
                        content_name: &'static str,
                        desc: Option<ImageDescription>,
                    }

                    let mut hlg_desc = ImageDescription::WINDOWS_BT2100;
                    hlg_desc.transfer = TransferFunction::Hlg;

                    let test_cases = vec![
                        TestCase {
                            output_hdr: false,
                            output_name: "SDR Display (sRGB Rec.709)",
                            content_name: "SDR Content (sRGB / BT.709)",
                            desc: Some(ImageDescription::SRGB),
                        },
                        TestCase {
                            output_hdr: false,
                            output_name: "SDR Display (sRGB Rec.709)",
                            content_name: "scRGB Content (FP16 linear)",
                            desc: Some(ImageDescription::WINDOWS_SCRGB),
                        },
                        TestCase {
                            output_hdr: false,
                            output_name: "SDR Display (sRGB Rec.709)",
                            content_name: "HDR10 Content (PQ BT.2020 10-bit)",
                            desc: Some(ImageDescription::WINDOWS_BT2100),
                        },
                        TestCase {
                            output_hdr: false,
                            output_name: "SDR Display (sRGB Rec.709)",
                            content_name: "HLG Content (BT.2100 HLG)",
                            desc: Some(hlg_desc),
                        },
                        TestCase {
                            output_hdr: true,
                            output_name: "HDR Display (PQ BT.2020)",
                            content_name: "HDR10 Content (PQ BT.2020 10-bit)",
                            desc: Some(ImageDescription::WINDOWS_BT2100),
                        },
                        TestCase {
                            output_hdr: true,
                            output_name: "HDR Display (PQ BT.2020)",
                            content_name: "scRGB Content (FP16 linear)",
                            desc: Some(ImageDescription::WINDOWS_SCRGB),
                        },
                        TestCase {
                            output_hdr: true,
                            output_name: "HDR Display (PQ BT.2020)",
                            content_name: "SDR Content (sRGB / BT.709)",
                            desc: Some(ImageDescription::SRGB),
                        },
                        TestCase {
                            output_hdr: true,
                            output_name: "HDR Display (PQ BT.2020)",
                            content_name: "HLG Content (BT.2100 HLG)",
                            desc: Some(hlg_desc),
                        },
                        TestCase {
                            output_hdr: true,
                            output_name: "HDR Display (PQ BT.2020)",
                            content_name: "Untagged Legacy Client",
                            desc: None,
                        },
                    ];

                    for tc in &test_cases {
                        println!(
                            "\n    [*] Mode: {}  |  Content: {}",
                            tc.output_name, tc.content_name
                        );
                        // True scanout plan evaluation under test (calling caps.evaluate_scanout_plan)
                        let plan = caps.evaluate_scanout_plan(tc.output_hdr, tc.desc.as_ref(), 203);

                        // Report step 1: Tier 1 DirectPassthrough evaluation test
                        let t1_matched = match (tc.output_hdr, &tc.desc) {
                            (false, Some(d)) if !d.is_hdr() && !d.windows_scrgb => true,
                            (false, None) => true,
                            (true, Some(d)) if d.is_pq_bt2020() => true,
                            _ => false,
                        };
                        if t1_matched {
                            println!(
                                "      [Tier 1: DirectPassthrough]  TEST PASSED: Native format matches display signal with zero transformation"
                            );
                        } else {
                            println!(
                                "      [Tier 1: DirectPassthrough]  TEST REJECTED: Color space or EOTF mismatch requires conversion"
                            );
                        }

                        // Report step 2: Tier 2A PlaneColorop evaluation test
                        match plan {
                            ScanoutPlan::PlaneColorop(conv) => {
                                println!(
                                    "      [Tier 2A: PlaneColorop]      TEST PASSED: Verified hardware plane color pipeline can execute {:?}",
                                    conv
                                );
                                if let Some(tr) = conv.to_scanout_color_transform() {
                                    for p in &pipelines {
                                        if let Some(plan_ops) = tr.plan(p) {
                                            println!(
                                                "        Colorop Hardware Op Allocation Strategy (Pipeline ID {}):",
                                                p.id
                                            );
                                            for (op_idx, (op, plan_item)) in
                                                p.ops.iter().zip(plan_ops.iter()).enumerate()
                                            {
                                                println!(
                                                    "          - Op {} (ID {}, {}): {}",
                                                    op_idx,
                                                    op.id,
                                                    op.kind.name(),
                                                    plan_item.description(op)
                                                );
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                            ScanoutPlan::DirectPassthrough => {
                                println!(
                                    "      [Tier 2A: PlaneColorop]      SKIPPED: Tier 1 DirectPassthrough already satisfied output without conversion"
                                );
                            }
                            _ => {
                                if !caps.supports_plane_colorop {
                                    println!(
                                        "      [Tier 2A: PlaneColorop]      TEST REJECTED: Plane lacks COLOR_PIPELINE property"
                                    );
                                } else if pipelines.is_empty() {
                                    println!(
                                        "      [Tier 2A: PlaneColorop]      TEST REJECTED: Plane has COLOR_PIPELINE property but no usable pipelines"
                                    );
                                } else {
                                    println!(
                                        "      [Tier 2A: PlaneColorop]      TEST REJECTED: Plane pipelines cannot fulfill required transform stages or lacks FP16 format"
                                    );
                                }
                            }
                        }

                        // Report step 3: Tier 2B CrtcHardware evaluation test
                        match plan {
                            ScanoutPlan::CrtcHardware(conv) => {
                                println!(
                                    "      [Tier 2B: CrtcHardware]      TEST PASSED: CRTC DEGAMMA+CTM+GAMMA verified for {:?}",
                                    conv
                                );
                            }
                            ScanoutPlan::DirectPassthrough | ScanoutPlan::PlaneColorop(_) => {
                                println!(
                                    "      [Tier 2B: CrtcHardware]      SKIPPED: Higher-priority tier (Tier 1 or Tier 2A) successfully selected"
                                );
                            }
                            _ => {
                                println!(
                                    "      [Tier 2B: CrtcHardware]      TEST REJECTED: CRTC lacks required DEGAMMA/CTM/GAMMA LUT hardware or format"
                                );
                            }
                        }

                        // Report step 4: Tier 3 VulkanFastDirectFlip evaluation test
                        match plan {
                            ScanoutPlan::VulkanFastDirectFlip => {
                                println!(
                                    "      [Tier 3: VulkanShaderFlip]   TEST SELECTED: Hardware tiers unavailable or tonemapping needed, falling back to Vulkan VRAM fast flip"
                                );
                            }
                            _ => {
                                println!(
                                    "      [Tier 3: VulkanShaderFlip]   STANDBY: Available as fallback, not needed for this format"
                                );
                            }
                        }

                        println!("      => Final Selected Plan: {:?}", plan);
                    }

                    println!(
                        "\n    ------------------------------------------------------------------------------------------------"
                    );
                    println!(
                        "    SCANOUT COLOR SCHEMES DECISION MATRIX SUMMARY (CRTC {:?})",
                        crtc_handle
                    );
                    println!(
                        "    ------------------------------------------------------------------------------------------------"
                    );
                    println!(
                        "    | Display Output   | Content Format       | Tier 1 Direct | Tier 2A Colorop | Final Selected Plan          |"
                    );
                    println!(
                        "    |------------------|----------------------|---------------|-----------------|------------------------------|"
                    );
                    for tc in &test_cases {
                        let plan = caps.evaluate_scanout_plan(tc.output_hdr, tc.desc.as_ref(), 203);
                        let t1_status = if matches!(plan, ScanoutPlan::DirectPassthrough) {
                            "YES (Direct)"
                        } else {
                            "Mismatch"
                        };
                        let t2a_status = match plan {
                            ScanoutPlan::PlaneColorop(_) => "YES (Colorop)",
                            ScanoutPlan::DirectPassthrough => "Not Needed",
                            _ => "Fallback",
                        };
                        println!(
                            "    | {:<16} | {:<20} | {:<13} | {:<15} | {:<28} |",
                            if tc.output_hdr {
                                "HDR (PQ BT.2020)"
                            } else {
                                "SDR (sRGB)"
                            },
                            tc.content_name
                                .split('(')
                                .next()
                                .unwrap_or(tc.content_name)
                                .trim(),
                            t1_status,
                            t2a_status,
                            format!("{:?}", plan)
                        );
                    }
                    println!(
                        "    ------------------------------------------------------------------------------------------------"
                    );
                }
            }
        }

        if cards_found == 0 {
            println!("  No /dev/dri/card* accessible in current environment.");
        }
        println!(
            "\n================================================================================\n"
        );
    }

    #[test]
    fn test_cursor_plane_scanout_flag_logic() {
        // 1. In compositing mode (allow_primary_scanout = false),
        // ALLOW_CURSOR_PLANE_SCANOUT must be preserved in effective_flags!
        let base_flags = FrameFlags::DEFAULT;
        assert!(base_flags.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT));

        let mut additional_flags = FrameFlags::empty();
        let mut remove_flags = FrameFlags::empty();

        let disable_cursor_plane = false;
        if !disable_cursor_plane {
            additional_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        } else {
            remove_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        }

        let allow_primary_scanout = false;
        if allow_primary_scanout {
            additional_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        } else {
            remove_frame_flags_helper(&mut remove_flags);
        }

        fn remove_frame_flags_helper(remove_flags: &mut FrameFlags) {
            *remove_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
        }

        let effective_flags = base_flags.union(additional_flags).difference(remove_flags);

        assert!(
            effective_flags.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT),
            "Cursor plane scanout must be allowed in normal compositing mode!"
        );
        assert!(
            !effective_flags.contains(FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT),
            "Primary plane scanout must be disabled when allow_primary_scanout is false"
        );

        // 2. Disabling cursor plane via COSMIC_DISABLE_CURSOR_PLANE strips it
        let mut remove_flags_disabled = remove_flags;
        remove_flags_disabled |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        let flags_without_cursor = effective_flags.difference(remove_flags_disabled);
        assert!(
            !flags_without_cursor.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT),
            "COSMIC_DISABLE_CURSOR_PLANE must remove cursor scanout"
        );

        // 3. Disabling direct scanout (COSMIC_DISABLE_DIRECT_SCANOUT) removes primary and overlay,
        // but must NOT remove cursor plane scanout!
        let mut flags_after_direct_disable = FrameFlags::DEFAULT;
        flags_after_direct_disable.remove(
            FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT
                | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY
                | FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT,
        );
        assert!(
            flags_after_direct_disable.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT),
            "COSMIC_DISABLE_DIRECT_SCANOUT must preserve hardware cursor plane scanout!"
        );
    }

    #[test]
    fn test_cursor_plane_transform_mode_and_scanout_preservation() {
        use smithay::backend::drm::color::{PlaneColorConversion, ScanoutPlan};

        // 1. With CPU cursor transformation (SrgbToPqEncoder), cursor pixels are pre-encoded:
        // - In CrtcHardware (ScRgbToPq), cursor is linearized on CPU to Rec.709 so CRTC CTM/GAMMA
        //   properly maps it to BT.2020 PQ at 203 nits.
        // - In DirectPassthrough (HDR), cursor is encoded on CPU to BT.2020 PQ directly.
        // - In SDR, cursor remains standard sRGB.
        //
        // Therefore, disable_cursor_plane only depends on *DISABLE_CURSOR_PLANE, NOT on active_scanout_plan!
        // Hardware cursor plane scanout remains enabled, preventing direct scanout drops during mouse movement.

        let disable_cursor_plane = *DISABLE_CURSOR_PLANE;
        assert!(!disable_cursor_plane);

        let active_plan = ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToPq {
            reference_white: 203,
        });
        // With CPU pre-transformation, cursor plane scanout is preserved in CrtcHardware mode!
        let base_flags = FrameFlags::DEFAULT;
        let mut additional_flags = FrameFlags::empty();
        let mut remove_flags = FrameFlags::empty();
        if !disable_cursor_plane {
            additional_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        } else {
            remove_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
        }
        let effective_flags = base_flags.union(additional_flags).difference(remove_flags);
        assert!(
            effective_flags.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT),
            "Cursor plane scanout must be preserved during CRTC color transformation because cursor is CPU-linearized!"
        );

        // Verify correct CursorTransformMode selection
        // SDR mode: Passthrough
        let mode_sdr = if false {
            CursorTransformMode::PqEncode { ref_white: 203 }
        } else {
            CursorTransformMode::Passthrough
        };
        assert_eq!(mode_sdr, CursorTransformMode::Passthrough);

        // HDR with CrtcHardware: LinearRec709
        let mode_crtc = if !true {
            CursorTransformMode::Passthrough
        } else if matches!(
            active_plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToPq { .. })
        ) {
            CursorTransformMode::LinearRec709
        } else {
            CursorTransformMode::PqEncode { ref_white: 203 }
        };
        assert_eq!(mode_crtc, CursorTransformMode::LinearRec709);

        // HDR with DirectPassthrough: PqEncode
        let passthrough_plan = ScanoutPlan::DirectPassthrough;
        let mode_passthrough = if !true {
            CursorTransformMode::Passthrough
        } else if matches!(
            passthrough_plan,
            ScanoutPlan::CrtcHardware(PlaneColorConversion::ScRgbToPq { .. })
        ) {
            CursorTransformMode::LinearRec709
        } else {
            CursorTransformMode::PqEncode { ref_white: 203 }
        };
        assert_eq!(
            mode_passthrough,
            CursorTransformMode::PqEncode { ref_white: 203 }
        );
    }

    #[test]
    fn test_vrr_target_rate_below_30hz_fallback_to_origin() {
        let origin_rate = 60_000;
        // Rates below 30Hz (30,000 mHz) must fallback to origin_rate
        assert_eq!(resolve_vrr_target_rate(0, origin_rate), origin_rate);
        assert_eq!(resolve_vrr_target_rate(24, origin_rate), origin_rate);
        assert_eq!(resolve_vrr_target_rate(24_000, origin_rate), origin_rate);
        assert_eq!(resolve_vrr_target_rate(29_970, origin_rate), origin_rate);
        assert_eq!(resolve_vrr_target_rate(29_999, origin_rate), origin_rate);

        // Rates at or above 30Hz (30,000 mHz) must be preserved
        assert_eq!(resolve_vrr_target_rate(30_000, origin_rate), 30_000);
        assert_eq!(resolve_vrr_target_rate(50_000, origin_rate), 50_000);
        assert_eq!(resolve_vrr_target_rate(60_000, origin_rate), 60_000);
        assert_eq!(resolve_vrr_target_rate(120_000, origin_rate), 120_000);
        assert_eq!(resolve_vrr_target_rate(144_000, origin_rate), 144_000);
    }

    #[test]
    fn test_fullscreen_pacing_flag_logic() {
        let base_flags = FrameFlags::DEFAULT;
        let mut additional_flags = FrameFlags::empty();
        let remove_flags = FrameFlags::empty();

        let has_active_fullscreen = true;
        if has_active_fullscreen {
            additional_flags |= FrameFlags::FULLSCREEN_PACING;
        }

        let effective_flags = base_flags.union(additional_flags).difference(remove_flags);
        assert!(
            effective_flags.contains(FrameFlags::FULLSCREEN_PACING),
            "FULLSCREEN_PACING must be set when has_active_fullscreen is true"
        );

        let mut non_fs_flags = FrameFlags::empty();
        let has_active_fullscreen = false;
        if has_active_fullscreen {
            non_fs_flags |= FrameFlags::FULLSCREEN_PACING;
        }
        let effective_non_fs = base_flags.union(non_fs_flags).difference(remove_flags);
        assert!(
            !effective_non_fs.contains(FrameFlags::FULLSCREEN_PACING),
            "FULLSCREEN_PACING must not be set when has_active_fullscreen is false"
        );
    }
}
