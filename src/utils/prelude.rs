use cosmic_comp_config::output::comp::{AdaptiveSync, OutputConfig, OutputState};
use parking_lot::RwLock;
use smithay::{
    backend::{
        allocator::{Buffer as _, format::is_ycbcr},
        drm::{
            DrmScanoutCapabilities, ScanoutCandidate, ScanoutPlan, ScanoutTarget,
            VrrSupport as Support,
        },
        renderer::utils::RendererSurfaceStateUserData,
    },
    desktop::utils::with_surfaces_surface_tree,
    output::{Output, WeakOutput},
    reexports::wayland_server::{Client, protocol::wl_surface::WlSurface},
    utils::Rectangle,
    wayland::{
        color::management::{ImageDescription, surface_description_from_states},
        compositor::{Barrier, BufferAssignment, CompositorHandler, SurfaceAttributes},
        dmabuf::get_dmabuf,
        seat::WaylandFocus,
        tearing_control::prefer_async_from_states,
    },
};

pub use super::geometry::*;
pub use crate::shell::{SeatExt, Shell, Workspace};
pub use crate::state::{Common, State};
pub use crate::wayland::handlers::xdg_shell::popup::update_reactive_popups;
use crate::{
    backend::kms::drm_helpers::HdrOutputState,
    config::EdidProduct,
    shell::{CosmicSurface, element::surface::WeakCosmicSurface, zoom::OutputZoomState},
    utils::env,
};

use std::{
    cell::{Ref, RefCell, RefMut},
    sync::{
        Mutex,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

pub trait OutputExt {
    fn is_internal(&self) -> bool;
    fn geometry(&self) -> Rectangle<i32, Global>;
    fn zoomed_geometry(&self) -> Option<Rectangle<i32, Global>>;

    fn adaptive_sync(&self) -> AdaptiveSync;
    fn set_adaptive_sync(&self, vrr: AdaptiveSync);
    fn adaptive_sync_support(&self) -> Option<Support>;
    fn set_adaptive_sync_support(&self, vrr: Option<Support>);
    fn vrr_target_rate(&self) -> Option<u32>;
    fn set_vrr_target_rate(&self, rate: Option<u32>);
    fn mirroring(&self) -> Option<Output>;
    fn set_mirroring(&self, output: Option<Output>);

    fn is_enabled(&self) -> bool;
    fn config(&self) -> Ref<'_, OutputConfig>;
    fn config_mut(&self) -> RefMut<'_, OutputConfig>;

    fn edid(&self) -> Option<&EdidProduct>;

    fn fifo_barrier(&self, barrier: Barrier, surface: WlSurface, client: Client);
    fn signal_fifo(&self, state: &mut State);

    fn set_avg_frametime(&self, duration: Option<Duration>);
    fn get_avg_frametime(&self) -> Option<Duration>;

    fn set_scanout_capabilities(&self, caps: DrmScanoutCapabilities);
    fn scanout_capabilities(&self) -> Option<DrmScanoutCapabilities>;

    fn set_fullscreen_occupied(&self, occupied: Option<FullscreenOccupied>);
    fn is_foreground_fullscreen_occupied(&self) -> Option<FullscreenOccupied>;
    fn refresh_fullscreen_occupied_flags(&self);
    fn set_fullscreen_scanout_plan(&self, plan: ScanoutPlan);
}

struct Vrr(AtomicU8);
struct VrrSupport(AtomicU8);
struct Mirroring(Mutex<Option<WeakOutput>>);
struct OutputScanoutCapabilities(RwLock<Option<DrmScanoutCapabilities>>);

#[derive(Debug, Clone, PartialEq)]
pub struct FullscreenOccupied {
    pub surface: CosmicSurface,
    pub prefers_async: bool,
    pub is_hdr: bool,
    pub is_yuv: bool,
    pub color_description: Option<ImageDescription>,
    pub scanout_plan: ScanoutPlan,
}

impl std::ops::Deref for FullscreenOccupied {
    type Target = CosmicSurface;

    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

impl FullscreenOccupied {
    pub fn new(
        surface: CosmicSurface,
        prefers_async: bool,
        is_hdr: bool,
        color_description: Option<ImageDescription>,
    ) -> Self {
        let is_yuv = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_is_yuv);
        Self {
            surface,
            prefers_async,
            is_hdr,
            is_yuv,
            color_description,
            scanout_plan: ScanoutPlan::DirectPassthrough,
        }
    }

    #[inline]
    pub fn effective_scanout_plan(&self) -> ScanoutPlan {
        self.scanout_plan
    }

    #[inline]
    pub fn tearing(&self) -> bool {
        self.prefers_async
    }

    #[inline]
    pub fn is_hdr(&self) -> bool {
        self.is_hdr
    }
}

#[derive(Debug, Clone)]
struct WeakFullscreenOccupied {
    surface: WeakCosmicSurface,
    prefers_async: bool,
    is_hdr: bool,
    is_yuv: bool,
    raw_color_description: Option<ImageDescription>,
    color_description: Option<ImageDescription>,
    cached_peak: Option<u32>,
    cached_ref_white: u32,
    scanout_plan: ScanoutPlan,
}

struct OutputFullscreenOccupied(RwLock<Option<WeakFullscreenOccupied>>);

#[derive(Debug, Clone)]
pub struct FifoBarrierItem {
    pub barrier: Barrier,
    pub surface: WlSurface,
    pub client: Client,
}

#[derive(Default)]
struct FifoBarriers(Mutex<Vec<FifoBarrierItem>>);

struct AvgFrameTime(RwLock<Option<Duration>>);

impl OutputExt for Output {
    fn is_internal(&self) -> bool {
        let name = self.name();
        name.starts_with("eDP-") || name.starts_with("LVDS-") || name.starts_with("DSI-")
    }

    fn geometry(&self) -> Rectangle<i32, Global> {
        Rectangle::new(self.current_location(), {
            self.current_transform()
                .transform_size(
                    self.current_mode()
                        .map(|m| m.size)
                        .unwrap_or_else(|| (0, 0).into()),
                )
                .to_f64()
                .to_logical(self.current_scale().fractional_scale())
                .to_i32_round()
        })
        .as_global()
    }

    fn zoomed_geometry(&self) -> Option<Rectangle<i32, Global>> {
        let output_geometry = self.geometry();

        let output_state = self.user_data().get::<Mutex<OutputZoomState>>()?;
        let mut output_state_ref = output_state.lock().unwrap();

        let focal_point = output_state_ref.current_focal_point().to_global(self);
        let mut zoomed_output_geo = output_geometry.to_f64();
        zoomed_output_geo.loc -= focal_point;
        zoomed_output_geo = zoomed_output_geo.downscale(output_state_ref.current_level());
        zoomed_output_geo.loc += focal_point;

        Some(zoomed_output_geo.to_i32_round())
    }

    fn adaptive_sync(&self) -> AdaptiveSync {
        self.user_data()
            .get::<Vrr>()
            .map(|vrr| match vrr.0.load(Ordering::SeqCst) {
                2 => AdaptiveSync::Force,
                1 => AdaptiveSync::Enabled,
                _ => AdaptiveSync::Disabled,
            })
            .unwrap_or(AdaptiveSync::Disabled)
    }
    fn set_adaptive_sync(&self, vrr: AdaptiveSync) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| Vrr(AtomicU8::new(0)));
        user_data.get::<Vrr>().unwrap().0.store(
            match vrr {
                AdaptiveSync::Disabled => 0,
                AdaptiveSync::Enabled => 1,
                AdaptiveSync::Force => 2,
            },
            Ordering::SeqCst,
        );
    }

    fn adaptive_sync_support(&self) -> Option<Support> {
        self.user_data()
            .get::<VrrSupport>()
            .and_then(|vrr| match vrr.0.load(Ordering::SeqCst) {
                0 => None,
                2 => Some(Support::RequiresModeset),
                3 => Some(Support::Supported),
                _ => Some(Support::NotSupported),
            })
    }

    fn set_adaptive_sync_support(&self, vrr: Option<Support>) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| VrrSupport(AtomicU8::new(0)));
        user_data.get::<VrrSupport>().unwrap().0.store(
            match vrr {
                None => 0,
                Some(Support::NotSupported) => 1,
                Some(Support::RequiresModeset) => 2,
                Some(Support::Supported) => 3,
            },
            Ordering::SeqCst,
        );
    }

    fn vrr_target_rate(&self) -> Option<u32> {
        self.config().vrr_target_rate
    }

    fn set_vrr_target_rate(&self, rate: Option<u32>) {
        self.config_mut().vrr_target_rate = rate;
    }

    fn mirroring(&self) -> Option<Output> {
        self.user_data().get::<Mirroring>().and_then(|mirroring| {
            mirroring
                .0
                .lock()
                .unwrap()
                .clone()
                .and_then(|o| o.upgrade())
        })
    }
    fn set_mirroring(&self, output: Option<Output>) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| Mirroring(Mutex::new(None)));
        *user_data.get::<Mirroring>().unwrap().0.lock().unwrap() =
            output.map(|output| output.downgrade());
    }

    fn is_enabled(&self) -> bool {
        self.user_data()
            .get::<RefCell<OutputConfig>>()
            .map(|conf| conf.borrow().enabled != OutputState::Disabled)
            .unwrap_or(false)
    }

    fn config(&self) -> Ref<'_, OutputConfig> {
        self.user_data()
            .get::<RefCell<OutputConfig>>()
            .unwrap()
            .borrow()
    }

    fn config_mut(&self) -> RefMut<'_, OutputConfig> {
        self.user_data()
            .get::<RefCell<OutputConfig>>()
            .unwrap()
            .borrow_mut()
    }

    fn edid(&self) -> Option<&EdidProduct> {
        self.user_data().get()
    }

    fn fifo_barrier(&self, barrier: Barrier, surface: WlSurface, client: Client) {
        self.user_data()
            .get_or_insert_threadsafe(|| FifoBarriers(Mutex::new(Vec::new())))
            .0
            .lock()
            .unwrap()
            .push(FifoBarrierItem {
                barrier,
                surface,
                client,
            });
    }

    fn signal_fifo(&self, state: &mut State) {
        let Some(fifo_barriers) = self.user_data().get::<FifoBarriers>() else {
            return;
        };

        let mut items = Vec::new();
        fifo_barriers.0.lock().unwrap().drain(..).for_each(|item| {
            item.barrier.signal();
            if !items.iter().any(|(s, _)| s == &item.surface) {
                items.push((item.surface, item.client));
            }
        });

        let dh = state.common.display_handle.clone();
        for (surface, client) in items {
            state
                .client_compositor_state(&client)
                .surface_blocker_cleared(&surface, state, &dh);
        }
    }

    fn set_avg_frametime(&self, duration: Option<Duration>) {
        *self
            .user_data()
            .get_or_insert_threadsafe(|| AvgFrameTime(RwLock::new(None)))
            .0
            .write() = duration;
    }

    fn get_avg_frametime(&self) -> Option<Duration> {
        *self.user_data().get::<AvgFrameTime>()?.0.read()
    }

    fn scanout_capabilities(&self) -> Option<DrmScanoutCapabilities> {
        let user_data = self.user_data();
        let guard = user_data.get::<OutputScanoutCapabilities>()?.0.read();
        guard.clone()
    }

    fn set_scanout_capabilities(&self, caps: DrmScanoutCapabilities) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| {
            OutputScanoutCapabilities(parking_lot::RwLock::new(None))
        });
        *user_data
            .get::<OutputScanoutCapabilities>()
            .unwrap()
            .0
            .write() = Some(caps);
    }

    fn set_fullscreen_occupied(&self, mut occupied: Option<FullscreenOccupied>) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| {
            OutputFullscreenOccupied(parking_lot::RwLock::new(None))
        });
        let lock = &user_data.get::<OutputFullscreenOccupied>().unwrap().0;
        let should_update = match (&*lock.read(), &occupied) {
            (Some(current), Some(next)) => {
                current.surface.upgrade().as_ref() != Some(&next.surface)
                    || current.is_hdr != next.is_hdr
                    || current.prefers_async != next.prefers_async
                    || current.color_description != next.color_description
            }
            (None, None) => false,
            _ => true,
        };
        if !should_update {
            return;
        }

        if let Some(ref mut occ) = occupied {
            // Check output HDR status, reference white, and peak luminance
            let (output_hdr_enabled, output_ref_white, output_peak) = user_data
                .get::<HdrOutputState>()
                .and_then(|s| s.get().or_else(|| s.staged()))
                .map(|hdr| {
                    (
                        true,
                        if hdr.reference_white > 0 {
                            hdr.reference_white
                        } else {
                            203
                        },
                        if hdr.capabilities.max_luminance > 0 {
                            Some(hdr.capabilities.max_luminance)
                        } else {
                            None
                        },
                    )
                })
                .unwrap_or((false, 203, None));

            // Query hardware scanout capabilities from output
            let caps = self.scanout_capabilities().unwrap_or_default();

            // Perform color inspection if not already set on occupied
            if occ.color_description.is_none() {
                if let Some(wl_surf) = occ.surface.wl_surface() {
                    occ.color_description = surface_tree_color_description(&wl_surf);
                }
            }

            let raw_desc = occ.color_description;

            if env::hdr_policy().remap_metadata && output_hdr_enabled {
                if let (Some(desc), Some(peak)) = (occ.color_description.as_ref(), output_peak) {
                    if desc.is_pq_bt2020() {
                        occ.color_description =
                            Some(desc.remap_metadata_to_peak(peak as u32, output_ref_white as u32));
                    }
                }
            }

            let is_yuv = occ
                .surface
                .wl_surface()
                .as_deref()
                .is_some_and(surface_tree_is_yuv);
            occ.is_yuv = is_yuv;

            // Determine scanout plan using smithay's unified evaluate_scanout
            let candidate = ScanoutCandidate {
                image_description: occ.color_description.as_ref(),
                is_yuv,
            };
            let target = ScanoutTarget {
                hdr_enabled: output_hdr_enabled,
                reference_white: output_ref_white,
                peak_luminance: output_peak,
            };
            let plan = caps.evaluate_scanout(candidate, target);
            tracing::info!(
                output = %self.name(),
                output_hdr = output_hdr_enabled,
                output_ref_white,
                output_peak = ?output_peak,
                color_desc = ?occ.color_description,
                is_yuv,
                ?plan,
                "Fullscreen occupied: evaluated hardware scanout plan"
            );

            occ.scanout_plan = plan;

            let cached_peak = output_peak.map(|p| p as u32);
            let cached_ref_white = output_ref_white as u32;

            *lock.write() = Some(WeakFullscreenOccupied {
                surface: occ.surface.downgrade(),
                prefers_async: occ.prefers_async,
                is_hdr: occ.is_hdr,
                is_yuv: occ.is_yuv,
                raw_color_description: raw_desc,
                color_description: occ.color_description,
                cached_peak,
                cached_ref_white,
                scanout_plan: occ.scanout_plan,
            });
        } else {
            *lock.write() = None;
        }
    }

    fn is_foreground_fullscreen_occupied(&self) -> Option<FullscreenOccupied> {
        let state = self.user_data().get::<OutputFullscreenOccupied>()?;
        let guard = state.0.read();
        let weak_occ = guard.as_ref()?;
        let surface = weak_occ.surface.upgrade()?;
        Some(FullscreenOccupied {
            surface,
            prefers_async: weak_occ.prefers_async,
            is_hdr: weak_occ.is_hdr,
            is_yuv: weak_occ.is_yuv,
            color_description: weak_occ.color_description.clone(),
            scanout_plan: weak_occ.scanout_plan,
        })
    }

    fn refresh_fullscreen_occupied_flags(&self) {
        let Some(state) = self.user_data().get::<OutputFullscreenOccupied>() else {
            return;
        };
        let (
            surface,
            current_async,
            current_raw_desc,
            current_is_yuv,
            current_peak,
            current_ref_white,
        ) = {
            let guard = state.0.read();
            let Some(weak_occ) = guard.as_ref() else {
                return;
            };
            let Some(surface) = weak_occ.surface.upgrade() else {
                return;
            };
            (
                surface,
                weak_occ.prefers_async,
                weak_occ.raw_color_description,
                weak_occ.is_yuv,
                weak_occ.cached_peak,
                weak_occ.cached_ref_white,
            )
        };
        let prefers_async = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_prefers_async);
        let raw_color_desc = surface
            .wl_surface()
            .as_deref()
            .and_then(surface_tree_color_description);
        let is_yuv = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_is_yuv);

        let (output_hdr_enabled, output_ref_white, output_peak) = self
            .user_data()
            .get::<HdrOutputState>()
            .and_then(|s| s.get().or_else(|| s.staged()))
            .map(|hdr| {
                (
                    true,
                    if hdr.reference_white > 0 {
                        hdr.reference_white
                    } else {
                        203
                    },
                    if hdr.capabilities.max_luminance > 0 {
                        Some(hdr.capabilities.max_luminance)
                    } else {
                        None
                    },
                )
            })
            .unwrap_or((false, 203, None));

        let peak_u32 = output_peak.map(|p| p as u32);
        let ref_white_u32 = output_ref_white as u32;

        if current_async == prefers_async
            && current_raw_desc == raw_color_desc
            && current_is_yuv == is_yuv
            && current_peak == peak_u32
            && current_ref_white == ref_white_u32
        {
            return;
        }

        let mut color_desc = raw_color_desc;
        if env::hdr_policy().remap_metadata && output_hdr_enabled {
            if let (Some(desc), Some(peak)) = (color_desc.as_ref(), output_peak) {
                if desc.is_pq_bt2020() {
                    color_desc =
                        Some(desc.remap_metadata_to_peak(peak as u32, output_ref_white as u32));
                }
            }
        }

        let caps = self.scanout_capabilities().unwrap_or_default();
        let candidate = ScanoutCandidate {
            image_description: color_desc.as_ref(),
            is_yuv,
        };
        let target = ScanoutTarget {
            hdr_enabled: output_hdr_enabled,
            reference_white: output_ref_white,
            peak_luminance: output_peak,
        };
        let plan = caps.evaluate_scanout(candidate, target);
        let mut guard = state.0.write();
        let Some(weak_occ) = guard.as_mut() else {
            return;
        };
        if weak_occ.surface.upgrade().as_ref() == Some(&surface) {
            weak_occ.prefers_async = prefers_async;
            weak_occ.is_yuv = is_yuv;
            weak_occ.raw_color_description = raw_color_desc;
            weak_occ.color_description = color_desc;
            weak_occ.cached_peak = peak_u32;
            weak_occ.cached_ref_white = ref_white_u32;
            weak_occ.scanout_plan = plan;
        }
    }

    fn set_fullscreen_scanout_plan(&self, plan: ScanoutPlan) {
        let Some(state) = self.user_data().get::<OutputFullscreenOccupied>() else {
            return;
        };
        let mut guard = state.0.write();
        if let Some(ref mut weak_occ) = *guard {
            weak_occ.scanout_plan = plan;
        }
    }
}

/// Whether any surface in the tree asked for tearing presentation via
/// `wp_tearing_control_v1`. Same locking rule as the description gates: read
/// from `states` inside the traversal, never re-lock the surface.
pub fn surface_tree_prefers_async(surface: &WlSurface) -> bool {
    let mut found = false;
    with_surfaces_surface_tree(surface, |_, states| {
        if prefer_async_from_states(states) {
            found = true;
        }
    });
    found
}

pub fn surface_tree_color_description(surface: &WlSurface) -> Option<ImageDescription> {
    let mut desc = None;
    with_surfaces_surface_tree(surface, |_, states| {
        if desc.is_none() {
            if let (Some(d), _) = surface_description_from_states(states) {
                desc = Some(d);
            }
        }
    });
    desc
}

pub fn surface_tree_is_hdr(surface: &WlSurface) -> bool {
    let mut found = false;
    with_surfaces_surface_tree(surface, |_, states| {
        if surface_description_from_states(states)
            .0
            .is_some_and(|description| description.is_pq_bt2020())
        {
            found = true;
        }
    });
    found
}

pub fn surface_tree_is_yuv(surface: &WlSurface) -> bool {
    let mut found = false;
    with_surfaces_surface_tree(surface, |_, states| {
        if found {
            return;
        }
        if let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() {
            if let Ok(guard) = data.lock() {
                if guard.is_ycbcr() {
                    found = true;
                    return;
                }
            }
        }
        let mut guard = states.cached_state.get::<SurfaceAttributes>();
        let attrs = guard.current();
        if let Some(BufferAssignment::NewBuffer(buffer)) = &attrs.buffer {
            if let Ok(dmabuf) = get_dmabuf(buffer) {
                if is_ycbcr(dmabuf.format().code) {
                    found = true;
                }
            }
        }
    });
    found
}
