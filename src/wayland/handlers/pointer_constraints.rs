// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::{CosmicSurface, WorkspaceSet},
    state::State,
    utils::prelude::*,
};
use smithay::{
    delegate_pointer_constraints,
    desktop::{layer_map_for_output, WindowSurfaceType},
    input::pointer::PointerHandle,
    output::Output,
    reexports::wayland_server::{protocol::wl_surface::WlSurface, Resource},
    utils::{Logical, Point, Rectangle},
    wayland::{
        compositor::CompositorHandler,
        pointer_constraints::PointerConstraintsHandler,
        seat::WaylandFocus,
    },
};

pub use smithay::wayland::pointer_constraints::with_pointer_constraint;

fn find_window<'a>(
    out: &'a Output,
    set: &WorkspaceSet,
    surface: &WlSurface,
) -> Option<(&'a Output, Rectangle<i32, Local>, Point<i32, Logical>)> {
    set.sticky_layer
        .mapped()
        .find_map(|w| {
            w.surface_offset(surface).and_then(|offset| {
                set.sticky_layer
                    .element_geometry(w)
                    .map(|geom| (out, geom, offset))
            })
        })
        .or_else(|| {
            set.workspaces.iter().find_map(|workspace| {
                workspace
                    .get_fullscreen()
                    .and_then(|fullscreen| {
                        fullscreen.surface_offset(surface).and_then(|offset| {
                            workspace
                                .fullscreen_geometry()
                                .map(|geom| (out, geom, offset))
                        })
                    })
                    .or_else(|| {
                        workspace.mapped().find_map(|w| {
                            w.surface_offset(surface).and_then(|offset| {
                                workspace
                                    .element_geometry(w)
                                    .map(|geom| (out, geom, offset))
                            })
                        })
                    })
            })
        })
        .or_else(|| {
            layer_map_for_output(out).layers().find_map(|l| {
                CosmicSurface::surface_tree_offset(l.wl_surface(), surface)
                    .map(|offset| (out, l.geometry().as_local(), offset))
            })
        })
}

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        // XXX region
        let seat = self
            .common
            .shell
            .read()
            .seats
            .iter()
            .find(|s| s.get_pointer().as_ref() == Some(pointer))
            .cloned();
        let focused = seat
            .and_then(|s| s.get_keyboard())
            .and_then(|k| k.current_focus());
        let is_focused = focused.is_some_and(|f| {
            let shell = self.common.shell.read();
            if let Some(fe) = shell.focused_element(&f) {
                fe.has_surface(surface, WindowSurfaceType::ALL)
            } else if let crate::shell::focus::target::KeyboardFocusTarget::Fullscreen(s) = f {
                s.has_surface(surface, WindowSurfaceType::ALL)
            } else if let Some(root) = f.wl_surface() {
                CosmicSurface::surface_tree_offset(&root, surface).is_some()
            } else {
                false
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
        surface: &WlSurface,
        pointer: &PointerHandle<Self>,
        location: Point<f64, Logical>,
    ) {
        // Apply the hint immediately if the constraint is active.
        if with_pointer_constraint(surface, pointer, |constraint| {
            constraint.is_some_and(|c| c.is_active())
        }) {
            apply_cursor_hint(self, surface, pointer, location);
        }
    }
}

pub fn apply_cursor_hint(
    state: &mut State,
    surface: &WlSurface,
    pointer: &PointerHandle<State>,
    mut location: Point<f64, Logical>,
) {
    if let Some(client) = surface.client() {
        let scale = state.client_compositor_state(&client).client_scale();
        location.x /= scale;
        location.y /= scale;
    }

    let point_and_output = {
        if let Some((out, geometry, surface_offset)) = state
            .common
            .shell
            .read()
            .workspaces
            .sets
            .iter()
            .find_map(|(out, set)| find_window(out, set, surface))
        {
            let pos_in_element = location + surface_offset.to_f64();
            let window_size = geometry.size.to_f64();

            if pos_in_element.x >= 0.0
                && pos_in_element.y >= 0.0
                && pos_in_element.x <= window_size.w
                && pos_in_element.y <= window_size.h
            {
                let origin = geometry.loc.to_f64();
                // the offset from the output (monitor position)
                let workspace_origin = out.geometry().loc.to_f64();
                let x = workspace_origin.x + origin.x + pos_in_element.x;
                let y = workspace_origin.y + origin.y + pos_in_element.y;
                Some((Point::new(x, y), out.clone()))
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some((point, output)) = point_and_output {
        let original_position = pointer.current_location();
        pointer.set_location(point);
        crate::write_point_position(point.x, point.y);

        let mut shell = state.common.shell.write();
        shell.update_pointer_position(point.as_global().to_local(&output), &output);

        let seat = shell
            .seats
            .iter()
            .find(|s| s.get_pointer().as_ref() == Some(pointer))
            .cloned();

        if let Some(seat) = seat {
            shell.update_focal_point(
                &seat,
                original_position.as_global(),
                state.common.config.cosmic_conf.accessibility_zoom.view_moves,
            );

            let output_geometry = output.geometry();
            for session in crate::input::cursor_sessions_for_output(&shell, &output) {
                if let Some((geometry, offset)) = seat.cursor_geometry(
                    point.to_buffer(
                        output.current_scale().fractional_scale(),
                        output.current_transform(),
                        &output_geometry.size.to_f64().as_logical(),
                    ),
                    state.common.clock.now(),
                ) {
                    if session
                        .current_constraints()
                        .map(|constraint| constraint.size != geometry.size)
                        .unwrap_or(true)
                    {
                        session.update_constraints(smithay::wayland::image_copy_capture::BufferConstraints {
                            size: geometry.size,
                            shm: vec![smithay::reexports::wayland_server::protocol::wl_shm::Format::Argb8888],
                            dma: None,
                        });
                    }
                    session.set_cursor_hotspot(offset);
                    session.set_cursor_pos(Some(geometry.loc));
                }
            }
        }
    }
}
delegate_pointer_constraints!(State);
