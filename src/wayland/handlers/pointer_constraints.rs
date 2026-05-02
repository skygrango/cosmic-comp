// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::WorkspaceSet,
    state::State,
    utils::prelude::{Local, OutputExt},
};
use smithay::{
    delegate_pointer_constraints,
    input::pointer::PointerHandle,
    output::Output,
    reexports::wayland_server::protocol::wl_surface::WlSurface,
    utils::{Logical, Point, Rectangle},
    wayland::{
        pointer_constraints::PointerConstraintsHandler,
        seat::WaylandFocus,
    },
};

pub use smithay::wayland::pointer_constraints::with_pointer_constraint;

fn find_window<'a>(
    out: &'a Output,
    set: &WorkspaceSet,
    surface: &WlSurface,
) -> Option<(
    &'a Output,
    Option<Point<i32, Logical>>,
    Rectangle<i32, Local>,
)> {
    set.sticky_layer
        .mapped()
        .find(|w| w.wl_surface().as_deref() == Some(surface))
        .and_then(|w| {
            set.sticky_layer
                .element_geometry(w)
                .map(|geom| (out, Some(w.active_window_offset()), geom))
        })
        .or_else(|| {
            set.workspaces.iter().find_map(|workspace| {
                workspace
                    .get_fullscreen()
                    .and_then(|fullscreen| {
                        if fullscreen.wl_surface().as_deref() == Some(surface) {
                            workspace
                                .fullscreen_geometry()
                                .map(|geom| (out, None, geom))
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        workspace
                            .mapped()
                            .find(|w| w.wl_surface().as_deref() == Some(surface))
                            .and_then(|w| {
                                workspace
                                    .element_geometry(w)
                                    .map(|geom| (out, Some(w.active_window_offset()), geom))
                            })
                    })
            })
        })
}

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        // XXX region
        let seat = self.common.shell.read().seats.iter().find(|s| s.get_pointer().as_ref() == Some(pointer)).cloned();
        let focused = seat.and_then(|s| s.get_keyboard()).and_then(|k| k.current_focus());
        let is_focused = focused.is_some_and(|f| {
            let shell = self.common.shell.read();
            if let Some(fe) = shell.focused_element(&f) {
                fe.has_surface(surface, smithay::desktop::WindowSurfaceType::ALL)
            } else if let crate::shell::focus::target::KeyboardFocusTarget::Fullscreen(s) = f {
                s.has_surface(surface, smithay::desktop::WindowSurfaceType::ALL)
            } else {
                f.wl_surface().as_deref() == Some(surface)
            }
        });

        if is_focused {
            with_pointer_constraint(surface, pointer, |constraint| {
                if let Some(constraint) = constraint {
                    constraint.activate();
                }
            });
        }
    }

    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: Point<f64, Logical>,
    ) {
        // Do nothing here. The hint is stored in the constraint by Smithay.
        // We will apply it when the constraint is deactivated.
    }
}

pub fn apply_cursor_hint(
    state: &mut State,
    surface: &WlSurface,
    pointer: &PointerHandle<State>,
    location: Point<f64, Logical>,
) {
    let point = {
        if let Some((out, header, geometry)) = state
            .common
            .shell
            .read()
            .workspaces
            .sets
            .iter()
            .find_map(|(out, set)| find_window(out, set, surface))
        {
            let window_size = geometry.size.to_f64();

            if location.x >= 0.0
                && location.y >= 0.0
                && location.x <= window_size.w
                && location.y <= window_size.h
            {
                let header_offset = header.map(|h| h.to_f64()).unwrap_or_default();
                let origin = geometry.loc.to_f64();
                // the offset from the output (monitor position)
                let workspace_origin = out.geometry().loc.to_f64();
                let x = workspace_origin.x + origin.x + header_offset.x + location.x;
                let y = workspace_origin.y + origin.y + header_offset.y + location.y;
                Some(Point::new(x, y))
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some(point) = point {
        pointer.set_location(point);
        crate::write_point_position(point.x, point.y);
    }
}
delegate_pointer_constraints!(State);
