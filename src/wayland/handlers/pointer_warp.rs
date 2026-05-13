// SPDX-License-Identifier: GPL-3.0-only

use crate::state::State;
use smithay::{
    delegate_pointer_warp,
    reexports::wayland_server::{
        Resource,
        protocol::{wl_pointer::WlPointer, wl_surface::WlSurface},
    },
    utils::{Logical, Point, Serial},
    wayland::pointer_warp::PointerWarpHandler,
};

impl PointerWarpHandler for State {
    fn warp_pointer(
        &mut self,
        surface: WlSurface,
        _pointer: WlPointer,
        pos: Point<f64, Logical>,
        serial: Serial,
    ) {
        let shell = self.common.shell.read();

        let pointer_handle = shell.seats.iter().find_map(|seat| {
            let pointer = seat.get_pointer()?;

            if pointer.last_enter() == Some(serial) {
                if let Some(focus) = pointer.current_focus() {
                    if let crate::shell::focus::target::PointerFocusTarget::WlSurface {
                        surface: focus_surface,
                        ..
                    } = focus
                    {
                        if focus_surface.id().same_client_as(&surface.id()) {
                            return Some(pointer.clone());
                        }
                    }
                }
            }

            None
        });

        drop(shell);

        if let Some(pointer_handle) = pointer_handle {
            // apply_cursor_hint maps the local surface coordinates to global
            // coordinates, applying constraints and preventing the pointer
            // from moving outside the window.
            self.apply_cursor_hint(&surface, &pointer_handle, pos);
        }
    }
}

delegate_pointer_warp!(State);
