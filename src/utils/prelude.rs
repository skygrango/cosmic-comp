use cosmic_comp_config::output::comp::{AdaptiveSync, OutputConfig, OutputState};
use parking_lot::RwLock;
use smithay::{
    backend::{
        drm::{DrmScanoutCapabilities, ScanoutPlan, VrrSupport as Support},
        renderer::utils::RendererSurfaceStateUserData,
    },
    desktop::utils::with_surfaces_surface_tree,
    output::{Output, WeakOutput},
    reexports::wayland_server::{Client, protocol::wl_surface::WlSurface},
    utils::{Physical, Rectangle, Size},
    wayland::{
        color::management::ImageDescription,
        compositor::{Barrier, CompositorHandler},
        seat::WaylandFocus,
        tearing_control::prefer_async_from_states,
    },
};

pub use super::geometry::*;
pub use crate::shell::{SeatExt, Shell, Workspace};
pub use crate::state::{Common, State};
pub use crate::wayland::handlers::xdg_shell::popup::update_reactive_popups;
use crate::{
    config::EdidProduct,
    shell::{CosmicSurface, element::surface::WeakCosmicSurface, zoom::OutputZoomState},
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

    fn scanout_capabilities(&self) -> Option<DrmScanoutCapabilities>;
    fn set_scanout_capabilities(&self, caps: DrmScanoutCapabilities);

    fn set_fullscreen_occupied(&self, occupied: Option<FullscreenOccupied>);
    fn is_foreground_fullscreen_occupied(&self) -> Option<FullscreenOccupied>;
    fn refresh_fullscreen_occupied_flags(&self);
    fn primary_fullscreen_surface(&self) -> Option<WlSurface>;
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
        Self {
            surface,
            prefers_async,
            is_hdr,
            color_description,
            scanout_plan: ScanoutPlan::DirectPassthrough,
        }
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
    color_description: Option<ImageDescription>,
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
                .get::<crate::backend::kms::drm_helpers::HdrOutputState>()
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

            // Determine scanout plan using smithay's evaluate_scanout_plan_with_peak
            let plan = caps.evaluate_scanout_plan_with_peak(
                output_hdr_enabled,
                occ.color_description.as_ref(),
                output_ref_white,
                output_peak,
            );

            tracing::info!(
                output = %self.name(),
                output_hdr = output_hdr_enabled,
                output_ref_white,
                output_peak = ?output_peak,
                color_desc = ?occ.color_description,
                ?plan,
                "Fullscreen occupied: evaluated hardware scanout plan"
            );

            occ.scanout_plan = plan;
        }

        *lock.write() = occupied.map(|occ| WeakFullscreenOccupied {
            surface: occ.surface.downgrade(),
            prefers_async: occ.prefers_async,
            is_hdr: occ.is_hdr,
            color_description: occ.color_description,
            scanout_plan: occ.scanout_plan,
        });
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
            color_description: weak_occ.color_description.clone(),
            scanout_plan: weak_occ.scanout_plan,
        })
    }

    fn refresh_fullscreen_occupied_flags(&self) {
        let Some(state) = self.user_data().get::<OutputFullscreenOccupied>() else {
            return;
        };
        let (surface, current_async, current_desc) = {
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
                weak_occ.color_description.clone(),
            )
        };
        let prefers_async = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_prefers_async);
        let color_desc = surface
            .wl_surface()
            .as_deref()
            .and_then(surface_tree_color_description);
        if current_async == prefers_async && current_desc == color_desc {
            return;
        }
        let (output_hdr_enabled, output_ref_white, output_peak) = self
            .user_data()
            .get::<crate::backend::kms::drm_helpers::HdrOutputState>()
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
        let caps = self.scanout_capabilities().unwrap_or_default();
        let plan = caps.evaluate_scanout_plan_with_peak(
            output_hdr_enabled,
            color_desc.as_ref(),
            output_ref_white,
            output_peak,
        );
        let mut guard = state.0.write();
        let Some(weak_occ) = guard.as_mut() else {
            return;
        };
        if weak_occ.surface.upgrade().as_ref() == Some(&surface) {
            weak_occ.prefers_async = prefers_async;
            weak_occ.color_description = color_desc;
            weak_occ.scanout_plan = plan;
        }
    }

    fn primary_fullscreen_surface(&self) -> Option<WlSurface> {
        let occupied = self.is_foreground_fullscreen_occupied()?;
        let root = occupied.surface.wl_surface()?;
        find_primary_fullscreen_surface(&root, self)
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
            if let (Some(d), _) =
                smithay::wayland::color::management::surface_description_from_states(states)
            {
                desc = Some(d);
            }
        }
    });
    desc
}

pub fn surface_tree_is_hdr(surface: &WlSurface) -> bool {
    let mut found = false;
    with_surfaces_surface_tree(surface, |_, states| {
        if smithay::wayland::color::management::surface_description_from_states(states)
            .0
            .is_some_and(|description| description.is_pq_bt2020())
        {
            found = true;
        }
    });
    found
}

/// Finds the primary fullscreen surface from a window's surface tree for the given output.
///
/// To qualify as the primary fullscreen surface:
/// 1. The surface must have an attached buffer.
/// 2. The surface/buffer size must closely match the screen/output size (within a 2-pixel
///    tolerance to account for integer rounding under scaling and avoid floating-point errors).
/// 3. If multiple surfaces match the screen size, the one that is largest and topmost in the
///    z-order (nearest to the screen) is selected.
///
/// If no surface covers the screen (e.g. undersized window or loading state), returns `None`.
pub fn find_primary_fullscreen_surface(root: &WlSurface, output: &Output) -> Option<WlSurface> {
    let output_logical = output.geometry().size;
    let output_mode = output
        .current_mode()
        .map(|m| output.current_transform().transform_size(m.size));
    let scale = output.current_scale().fractional_scale();

    // Candidates in top-to-bottom order (downward traversal visits topmost first)
    let mut candidates: Vec<(WlSurface, i64, usize)> = Vec::new();
    let mut depth: usize = 0;

    with_surfaces_surface_tree(root, |s, states| {
        depth += 1;
        let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() else {
            return;
        };
        let rstate = data.lock().unwrap();
        if rstate.buffer().is_none() {
            return;
        }

        let surf_logical = rstate.surface_size().or_else(|| rstate.buffer_size());
        let buf_scale = rstate.buffer_scale();
        let buf_dims: Option<Size<i32, Physical>> = rstate
            .buffer_size()
            .map(|sz| (sz.w * buf_scale, sz.h * buf_scale).into());

        // 1. Check logical size match within 2 pixels tolerance
        let logical_match = surf_logical.is_some_and(|sz| {
            (sz.w - output_logical.w).abs() <= 2 && (sz.h - output_logical.h).abs() <= 2
        });

        // 2. Check physical buffer match against output mode within 2 pixels
        let physical_match = if let (Some(buf), Some(mode)) = (buf_dims, output_mode) {
            (buf.w - mode.w).abs() <= 2 && (buf.h - mode.h).abs() <= 2
        } else {
            false
        };

        // 3. Check scaled logical match against physical mode within 2 pixels
        let scaled_match = if let (Some(sz), Some(mode)) = (surf_logical, output_mode) {
            let scaled_w = (sz.w as f64 * scale).round() as i32;
            let scaled_h = (sz.h as f64 * scale).round() as i32;
            (scaled_w - mode.w).abs() <= 2 && (scaled_h - mode.h).abs() <= 2
        } else {
            false
        };

        // 4. Check buffer size logical match
        let buf_logical_match = rstate.buffer_size().is_some_and(|bsz| {
            (bsz.w - output_logical.w).abs() <= 2 && (bsz.h - output_logical.h).abs() <= 2
        });

        if logical_match || physical_match || scaled_match || buf_logical_match {
            let area = if let Some(buf) = buf_dims {
                buf.w as i64 * buf.h as i64
            } else if let Some(sz) = surf_logical {
                sz.w as i64 * sz.h as i64
            } else {
                0
            };
            candidates.push((s.clone(), area, depth));
        }
    });

    if candidates.is_empty() {
        return None;
    }

    // Pick the largest candidate. If areas are roughly equal (both cover the screen),
    // pick the topmost one (smallest depth index in top-to-bottom traversal).
    candidates.sort_by(|a, b| {
        let area_diff = (a.1 - b.1).abs();
        let max_area = a.1.max(b.1);
        if max_area > 0 && (area_diff as f64 / max_area as f64) < 0.05 {
            a.2.cmp(&b.2)
        } else {
            b.1.cmp(&a.1)
        }
    });

    candidates.first().map(|(s, _, _)| s.clone())
}
