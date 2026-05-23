// SPDX-License-Identifier: GPL-3.0-only

use crate::state::State;
use smithay::{
    reexports::wayland_server::protocol::{wl_pointer::WlPointer, wl_surface::WlSurface},
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

        let pointer = shell.seats.iter().find_map(|seat| {
            if let Some(pointer) = seat.get_pointer()
                && pointer.last_enter() == Some(serial)
                && let Some(keyboard) = seat.get_keyboard()
                && let Some(keyboard_focus) = keyboard.current_focus()
                && keyboard_focus.has_surface(&shell, &surface)
            {
                return Some(pointer);
            }
            None
        });

        drop(shell);

        if let Some(pointer) = pointer {
            self.apply_cursor_hint(&surface, &pointer, pos);
        }
    }
}
