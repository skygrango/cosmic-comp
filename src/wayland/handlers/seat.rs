// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::{
        Devices,
        focus::target::{KeyboardFocusTarget, PointerFocusTarget},
    },
    state::State,
    utils::prelude::SeatExt,
    wayland::handlers::pointer_constraints::apply_cursor_hint,
};
use smithay::{
    delegate_cursor_shape, delegate_seat,
    input::{
        SeatHandler, SeatState,
        keyboard::LedState,
        pointer::{CursorImageStatus, PointerHandle},
    },
    reexports::wayland_server::protocol::wl_surface::WlSurface,
};

impl SeatHandler for State {
    type KeyboardFocus = KeyboardFocusTarget;
    type PointerFocus = PointerFocusTarget;
    type TouchFocus = PointerFocusTarget;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.common.seat_state
    }

    fn cursor_image(&mut self, seat: &smithay::input::Seat<Self>, image: CursorImageStatus) {
        seat.set_cursor_image_status(image);
    }

    fn focus_changed(
        &mut self,
        _seat: &smithay::input::Seat<Self>,
        _focused: Option<&Self::KeyboardFocus>,
    ) {
    }

    fn led_state_changed(&mut self, seat: &smithay::input::Seat<Self>, led_state: LedState) {
        let userdata = seat.user_data();
        let devices = userdata.get::<Devices>().unwrap();
        devices.update_led_state(led_state);
    }

    fn remove_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
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
}

delegate_seat!(State);
delegate_cursor_shape!(State);
