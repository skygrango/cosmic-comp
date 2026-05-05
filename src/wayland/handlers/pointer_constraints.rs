// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::{CosmicSurface, WorkspaceSet},
    state::State,
    utils::prelude::*,
};
use smithay::{
    delegate_pointer_constraints,
    desktop::{WindowSurfaceType, layer_map_for_output},
    input::pointer::PointerHandle,
    output::Output,
    reexports::wayland_server::{Resource, protocol::{wl_shm::Format, wl_surface::WlSurface}},
    utils::{Logical, Point, Rectangle},
    wayland::{
        compositor::CompositorHandler, image_copy_capture::BufferConstraints, pointer_constraints::PointerConstraintsHandler, seat::WaylandFocus
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
        let seat = self
            .common
            .shell
            .read()
            .seats
            .iter()
            .find(|s| s.get_pointer().as_ref() == Some(pointer))
            .cloned();

        let (is_under, is_focused, surface_location) = if let Some(seat) = seat {
            self.common.idle_notifier_state.notify_activity(&seat);
            let current_output = seat.active_output();
            let position = seat.get_pointer().unwrap().current_location().as_global();
            let shell = self.common.shell.read();

            let under =
                State::surface_under(position, &current_output, &shell).map(|(target, _)| target);
            let is_under = if let Some(under) = under
                && let Some(under_surface) = under.wl_surface()
            {
                *under_surface == *surface
                    || CosmicSurface::surface_tree_offset(surface, &under_surface).is_some()
            } else {
                false
            };

            let focused = seat.get_keyboard().and_then(|k| k.current_focus());
            let is_focused = focused.is_some_and(|f| {
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
            let surface_location = if is_under && is_focused {
                shell.workspaces.sets.iter().find_map(|(out, set)| {
                    find_window(out, set, surface).map(|(out, geometry, offset)| {
                        let out = out.geometry().loc.to_f64();
                        let geometry = geometry.loc.to_f64();
                        let offset = offset.to_f64();
                        let x = out.x + geometry.x + offset.x;
                        let y = out.y + geometry.y + offset.y;
                        Point::new(x, y)
                    })
                })
            } else {
                None
            };

            (is_under, is_focused, surface_location)
        } else {
            (false, false, None)
        };

        if is_under && is_focused {
            with_pointer_constraint(surface, pointer, |constraint| {
                if let Some(constraint) = constraint {
                    if let Some(region) = constraint.region() {
                        if let Some(surface_location) = surface_location {
                            let position = pointer.current_location();
                            let point = (position - surface_location).to_i32_round();
                            if region.contains(point) {
                                constraint.activate();
                            }
                        }
                    } else {
                        constraint.activate();
                    }
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
            let mut pos_in_element = location + surface_offset.to_f64();
            let window_size = geometry.size.to_f64();

            let is_legal = |p: Point<f64, Logical>| {
                p.x >= 0.0
                    && p.y >= 0.0
                    // hack: prevent the cursor from touching the edge of the window
                    && p.x <= window_size.w - 1.
                    && p.y <= window_size.h - 1.
            };

            if !is_legal(pos_in_element) {
                let original_global = pointer.current_location();
                let workspace_origin = out.geometry().loc.to_f64();
                let origin = geometry.loc.to_f64();

                let original_pos_in_element = Point::new(
                    original_global.x - workspace_origin.x - origin.x,
                    original_global.y - workspace_origin.y - origin.y,
                );

                let y_only_pos = Point::new(original_pos_in_element.x, pos_in_element.y);
                let x_only_pos = Point::new(pos_in_element.x, original_pos_in_element.y);

                if is_legal(y_only_pos) {
                    pos_in_element = y_only_pos;
                } else if is_legal(x_only_pos) {
                    pos_in_element = x_only_pos;
                } else {
                    pos_in_element = original_pos_in_element;
                }
            }

            let origin = geometry.loc.to_f64();
            let workspace_origin = out.geometry().loc.to_f64();
            let x = workspace_origin.x + origin.x + pos_in_element.x;
            let y = workspace_origin.y + origin.y + pos_in_element.y;
            Some((Point::new(x, y), out.clone()))
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
                state
                    .common
                    .config
                    .cosmic_conf
                    .accessibility_zoom
                    .view_moves,
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
                        session.update_constraints(BufferConstraints {
                            size: geometry.size,
                            shm: vec![Format::Argb8888],
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
