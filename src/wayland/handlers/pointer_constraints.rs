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
    reexports::wayland_server::{
        Resource,
        protocol::{wl_shm::Format, wl_surface::WlSurface},
    },
    utils::{Logical, Point, Rectangle},
    wayland::{
        compositor::{CompositorHandler, get_parent, with_states},
        image_copy_capture::BufferConstraints,
        pointer_constraints::{PointerConstraint, PointerConstraintsHandler},
        seat::WaylandFocus,
        shell::xdg::{XDG_POPUP_ROLE, XdgPopupSurfaceData},
    },
};

pub use smithay::wayland::pointer_constraints::{
    with_pointer_constraint, with_pointer_constraint_readonly,
};

fn find_window(
    output: &Output,
    set: &WorkspaceSet,
    surface: &WlSurface,
) -> Option<(Rectangle<i32, Local>, Point<i32, Logical>)> {
    let mut root = surface.clone();
    loop {
        while let Some(parent) = get_parent(&root) {
            root = parent;
        }
        if smithay::wayland::compositor::get_role(&root) == Some(XDG_POPUP_ROLE) {
            if let Some(parent) = with_states(&root, |states| {
                states
                    .data_map
                    .get::<XdgPopupSurfaceData>()
                    .and_then(|m| m.lock().unwrap().parent.as_ref().cloned())
            }) {
                root = parent;
                continue;
            }
        }
        break;
    }

    set.sticky_layer
        .mapped()
        .find(|w| {
            w.windows()
                .any(|(w, _)| w.wl_surface().as_deref() == Some(&root))
        })
        .and_then(|w| {
            w.surface_offset(surface).and_then(|offset| {
                set.sticky_layer
                    .element_geometry(w)
                    .map(|geom| (geom, offset))
            })
        })
        .or_else(|| {
            set.workspaces.iter().find_map(|workspace| {
                workspace
                    .get_fullscreen()
                    .and_then(|fullscreen| {
                        (fullscreen.wl_surface().as_deref() == Some(&root))
                            .then(|| {
                                fullscreen.surface_offset(surface).and_then(|offset| {
                                    workspace.fullscreen_geometry().map(|geom| (geom, offset))
                                })
                            })
                            .flatten()
                    })
                    .or_else(|| {
                        workspace.mapped().find_map(|w| {
                            w.windows()
                                .any(|(w, _)| w.wl_surface().as_deref() == Some(&root))
                                .then(|| {
                                    w.surface_offset(surface).and_then(|offset| {
                                        workspace.element_geometry(w).map(|geom| (geom, offset))
                                    })
                                })
                                .flatten()
                        })
                    })
            })
        })
        .or_else(|| {
            layer_map_for_output(output).layers().find_map(|l| {
                (l.wl_surface() == &root)
                    .then(|| {
                        CosmicSurface::surface_tree_offset(l.wl_surface(), surface)
                            .map(|offset| (l.geometry().as_local(), offset))
                    })
                    .flatten()
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
            seat.set_pointer_constraint_hint(None);
            self.common.idle_notifier_state.notify_activity(&seat);
            let current_output = seat.active_output();
            let position = seat.get_pointer().unwrap().current_location().as_global();
            let shell = self.common.shell.read();

            let under = State::surface_under(position, &current_output, &shell);
            let mut surface_location = None;
            let is_under = if let Some((target, target_loc)) = under
                && let Some(under_surface) = target.wl_surface()
            {
                if *under_surface == *surface {
                    surface_location = Some(target_loc);
                    true
                } else {
                    CosmicSurface::surface_tree_offset(surface, &under_surface).map_or(
                        false,
                        |offset| {
                            surface_location = Some(target_loc - offset.to_f64().as_global());
                            true
                        },
                    )
                }
            } else {
                false
            };

            let is_focused = seat
                .get_keyboard()
                .and_then(|k| k.current_focus())
                .is_some_and(|f| {
                    if let Some(fe) = shell.focused_element(&f) {
                        fe.has_surface(surface, WindowSurfaceType::ALL)
                    } else if let crate::shell::focus::target::KeyboardFocusTarget::Fullscreen(s) =
                        f
                    {
                        s.has_surface(surface, WindowSurfaceType::ALL)
                    } else if let Some(root) = f.wl_surface() {
                        CosmicSurface::surface_tree_offset(&root, surface).is_some()
                    } else {
                        false
                    }
                });

            (is_under, is_focused, surface_location)
        } else {
            (false, false, None)
        };

        if is_focused && is_under {
            with_pointer_constraint(self, surface, pointer, |constraint| {
                if let Some(mut constraint) = constraint {
                    if let Some(region) = constraint.region() {
                        if let Some(surface_location) = surface_location
                            && let position = pointer.current_location()
                            && let point = (position - surface_location.as_logical()).to_i32_round()
                            && region.contains(point)
                        {
                            constraint.activate();
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
        if with_pointer_constraint_readonly::<State, _, _>(surface, pointer, |constraint| {
            constraint.is_some_and(|c| c.is_active())
        }) {
            let seat = self
                .common
                .shell
                .read()
                .seats
                .iter()
                .find(|s| s.get_pointer().as_ref() == Some(pointer))
                .cloned();

            if let Some(seat) = seat {
                seat.set_pointer_constraint_hint(Some((surface.clone(), location)));
            }
        }
    }

    fn deactivated(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        let seat = self
            .common
            .shell
            .read()
            .seats
            .iter()
            .find(|s| s.get_pointer().as_ref() == Some(pointer))
            .cloned();

        if let Some(seat) = seat {
            if let Some((hint_surface, hint_location)) = seat.pointer_constraint_hint() {
                if hint_surface == *surface {
                    apply_cursor_hint(self, surface, pointer, hint_location);
                    seat.set_pointer_constraint_hint(None);
                }
            }
        }
    }

    fn activated(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {}
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
        let shell = state.common.shell.read();
        let found = shell.workspaces.sets.iter().find_map(|(out, set)| {
            find_window(out, set, surface)
                .map(|(geometry, surface_offset)| (out, geometry, surface_offset))
        });

        if let Some((output, geometry, surface_offset)) = found {
            let mut pos_in_element = location + surface_offset.to_f64();
            let window_size = geometry.size.to_f64();

            let is_legal = |p: Point<f64, Logical>| {
                let in_window = p.x >= 0.0
                    && p.y >= 0.0
                    // hack: prevent the cursor from touching the edge of the window
                    && p.x <= window_size.w - 1.
                    && p.y <= window_size.h - 1.;
                if !in_window {
                    return false;
                }

                with_pointer_constraint_readonly::<State, _, _>(surface, pointer, |constraint| {
                    if let Some(constraint) = constraint {
                        if let Some(region) = constraint.region() {
                            let point_in_surface = (p - surface_offset.to_f64()).to_i32_round();
                            return region.contains(point_in_surface);
                        }
                    }
                    true
                })
            };

            if !is_legal(pos_in_element) {
                let original_global = pointer.current_location();
                let workspace_origin = output.geometry().loc.to_f64();
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
            let workspace_origin = output.geometry().loc.to_f64();
            let x = workspace_origin.x + origin.x + pos_in_element.x;
            let y = workspace_origin.y + origin.y + pos_in_element.y;
            Some((Point::new(x, y), output.clone()))
        } else {
            None
        }
    };

    if let Some((point, output)) = point_and_output {
        let original_position = pointer.current_location();
        pointer.set_location(point);

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
