// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    shell::Devices,
    shell::focus::target::{KeyboardFocusTarget, PointerFocusTarget},
    state::State,
    utils::prelude::SeatExt,
};
use smithay::{
    input::{
        SeatHandler, SeatState,
        keyboard::LedState,
        pointer::{CursorImageStatus, PointerHandle},
    },
    reexports::wayland_server::protocol::wl_surface::WlSurface,
    wayland::pointer_constraints::PointerConstraint,
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

    fn remove_constraint(
        &mut self,
        surface: &WlSurface,
        pointer: &PointerHandle<Self>,
        constraint: Option<&PointerConstraint>,
    ) {
        let seat = self
            .common
            .shell
            .read()
            .seats
            .iter()
            .find(|s| s.get_pointer().as_ref() == Some(pointer))
            .cloned();

        if let Some(seat) = seat
            && let Some((hint_surface, hint_location)) = seat.pointer_constraint_hint()
            && hint_surface == *surface
        {
            self.apply_cursor_hint(surface, pointer, hint_location, constraint);
            seat.set_pointer_constraint_hint(None);
        }
    }
}
