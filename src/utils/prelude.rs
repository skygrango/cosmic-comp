use cosmic_comp_config::output::comp::{AdaptiveSync, OutputConfig, OutputState};
use parking_lot::RwLock;
use smithay::{
    backend::drm::VrrSupport as Support,
    output::{Output, WeakOutput},
    reexports::wayland_server::{Client, protocol::wl_surface::WlSurface},
    utils::Rectangle,
    wayland::{
        compositor::{Barrier, CompositorHandler},
        seat::WaylandFocus,
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

    fn set_fullscreen_occupied(&self, occupied: Option<FullscreenOccupied>);
    fn is_foreground_fullscreen_occupied(&self) -> Option<FullscreenOccupied>;
    fn refresh_fullscreen_occupied_flags(&self);
}

struct Vrr(AtomicU8);
struct VrrSupport(AtomicU8);
struct Mirroring(Mutex<Option<WeakOutput>>);

#[derive(Debug, Clone, PartialEq)]
pub struct FullscreenOccupied {
    pub surface: CosmicSurface,
    pub has_hdr: bool,
    pub prefers_async: bool,
}

impl std::ops::Deref for FullscreenOccupied {
    type Target = CosmicSurface;

    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

impl FullscreenOccupied {
    pub fn new(surface: CosmicSurface, has_hdr: bool, prefers_async: bool) -> Self {
        Self {
            surface,
            has_hdr,
            prefers_async,
        }
    }

    #[inline]
    pub fn hdr(&self) -> bool {
        self.has_hdr
    }

    #[inline]
    pub fn tearing(&self) -> bool {
        self.prefers_async
    }
}

#[derive(Debug, Clone)]
struct WeakFullscreenOccupied {
    surface: WeakCosmicSurface,
    has_hdr: bool,
    prefers_async: bool,
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

    fn set_fullscreen_occupied(&self, occupied: Option<FullscreenOccupied>) {
        let user_data = self.user_data();
        user_data.insert_if_missing_threadsafe(|| {
            OutputFullscreenOccupied(parking_lot::RwLock::new(None))
        });
        let lock = &user_data.get::<OutputFullscreenOccupied>().unwrap().0;
        let should_update = match (&*lock.read(), &occupied) {
            (Some(current), Some(next)) => {
                current.surface.upgrade().as_ref() != Some(&next.surface)
                    || current.has_hdr != next.has_hdr
                    || current.prefers_async != next.prefers_async
            }
            (None, None) => false,
            _ => true,
        };
        if !should_update {
            return;
        }
        *lock.write() = occupied.map(|occ| WeakFullscreenOccupied {
            surface: occ.surface.downgrade(),
            has_hdr: occ.has_hdr,
            prefers_async: occ.prefers_async,
        });
    }

    fn is_foreground_fullscreen_occupied(&self) -> Option<FullscreenOccupied> {
        let state = self.user_data().get::<OutputFullscreenOccupied>()?;
        let guard = state.0.read();
        let weak_occ = guard.as_ref()?;
        let surface = weak_occ.surface.upgrade()?;
        Some(FullscreenOccupied {
            surface,
            has_hdr: weak_occ.has_hdr,
            prefers_async: weak_occ.prefers_async,
        })
    }

    fn refresh_fullscreen_occupied_flags(&self) {
        let Some(state) = self.user_data().get::<OutputFullscreenOccupied>() else {
            return;
        };
        let mut guard = state.0.write();
        let Some(weak_occ) = guard.as_mut() else {
            return;
        };
        let Some(surface) = weak_occ.surface.upgrade() else {
            return;
        };
        let has_hdr = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_has_hdr_client_description);
        let prefers_async = surface
            .wl_surface()
            .as_deref()
            .is_some_and(surface_tree_prefers_async);
        weak_occ.has_hdr = has_hdr;
        weak_occ.prefers_async = prefers_async;
    }
}

/// Whether any surface in the tree asked for tearing presentation via
/// `wp_tearing_control_v1`. Same locking rule as the description gates: read
/// from `states` inside the traversal, never re-lock the surface.
pub fn surface_tree_prefers_async(surface: &WlSurface) -> bool {
    let mut found = false;
    smithay::desktop::utils::with_surfaces_surface_tree(surface, |_, states| {
        if smithay::wayland::tearing_control::prefer_async_from_states(states) {
            found = true;
        }
    });
    found
}

pub fn surface_tree_has_hdr_client_description(surface: &WlSurface) -> bool {
    let mut found = false;
    // The traversal callback runs with each surface's state lock held, so the
    // description must be read from `states`, never via `get_surface_description`.
    smithay::desktop::utils::with_surfaces_surface_tree(surface, |_, states| {
        if smithay::wayland::color::management::surface_description_from_states(states)
            .0
            .is_some_and(|description| description.is_hdr())
        {
            found = true;
        }
    });
    found
}
