// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::WorkspaceSet,
    state::State,
    utils::{
        geometry,
        prelude::{Local, OutputExt},
    },
    wayland::handlers::workspace,
};
use smithay::{
    delegate_pointer_constraints,
    input::pointer::PointerHandle,
    output::Output,
    reexports::wayland_server::protocol::wl_surface::WlSurface,
    utils::{Logical, Point, Rectangle},
    wayland::{
        pointer_constraints::{PointerConstraintsHandler, with_pointer_constraint},
        seat::WaylandFocus,
    },
};

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
        if pointer
            .current_focus()
            .is_some_and(|x| x.wl_surface().as_deref() == Some(surface))
        {
            with_pointer_constraint(surface, pointer, |constraint| {
                constraint.unwrap().activate();
                self.last_pointer = Some(pointer.current_location());
            });
        }else{
            self.last_pointer = None;
        }
    }

    fn cursor_position_hint(
        &mut self,
        surface: &WlSurface,
        pointer: &PointerHandle<Self>,
        location: Point<f64, Logical>,
    ) {
        if with_pointer_constraint(surface, pointer, |constraint| {
            constraint.is_some_and(|c| c.is_active())
        }) {
            if let Some((out, header, geometry)) = self
                .common
                .shell
                .read()
                .workspaces
                .sets
                .iter()
                .find_map(|(out, set)| find_window(out, set, surface))
            {
                let window_size = geometry.size.to_f64();
                let last_pointer = self.last_pointer.unwrap_or(Point::new(0.0, 0.0));
                if last_pointer.x >= 0.0
                    && last_pointer.y >= 0.0
                    && last_pointer.x <= window_size.w
                    && last_pointer.y <= window_size.h
                {
                    //let header_offset = header.map(|h| h.to_f64()).unwrap_or_default();
                    let origin = geometry.loc.to_f64();
                    // the offset from the output (monitor position)
                    let workspace_origin = out.geometry().loc.to_f64();

                    pointer.set_location(Point::new(
                        workspace_origin.x + last_pointer.x,
                        workspace_origin.y + last_pointer.y,
                    ));
                }
            };
        }
    }
}
delegate_pointer_constraints!(State);
