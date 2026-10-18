// SPDX-License-Identifier: GPL-3.0-only

use std::{
    cell::RefCell,
    os::unix::{
        io::{AsFd, OwnedFd},
        net::UnixStream,
    },
    sync::Arc,
    time::Duration,
};

use cosmic_comp_config::{
    XwaylandDescaling,
    output::comp::{AdaptiveSync, OutputConfig, OutputState, TransformDef},
};
use smithay::{
    backend::input::{
        AbsolutePositionEvent, ButtonState, Device, DeviceCapability, Event as InputBackendEvent,
        InputBackend, InputEvent, InputTime, PointerButtonEvent, PointerMotionAbsoluteEvent,
        PointerMotionEvent, UnusedEvent,
    },
    desktop::layer_map_for_output,
    output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel},
    reexports::{
        calloop::EventLoop,
        wayland_server::{Client, Display, Resource},
        x11rb::{
            connection::Connection as _,
            protocol::xproto::{ConnectionExt as _, CreateWindowAux, EventMask, WindowClass},
        },
    },
    utils::{Logical, Point, Rectangle, SERIAL_COUNTER, Size, Transform},
    wayland::seat::WaylandFocus,
};
use wayland_client::{
    Connection as ClientConnection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum,
    backend::ObjectId,
    protocol::{
        wl_buffer::WlBuffer,
        wl_callback::WlCallback,
        wl_compositor::WlCompositor,
        wl_pointer::{self, WlPointer},
        wl_region::WlRegion,
        wl_registry::{self, WlRegistry},
        wl_seat::{self, WlSeat},
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
        wl_subcompositor::WlSubcompositor,
        wl_subsurface::WlSubsurface,
        wl_surface::WlSurface as ClientWlSurface,
    },
};
use wayland_protocols::{
    wp::{
        pointer_constraints::zv1::client::{
            zwp_confined_pointer_v1::{self, ZwpConfinedPointerV1},
            zwp_locked_pointer_v1::{self, ZwpLockedPointerV1},
            zwp_pointer_constraints_v1::{self, ZwpPointerConstraintsV1},
        },
        pointer_warp::v1::client::wp_pointer_warp_v1::WpPointerWarpV1,
        relative_pointer::zv1::client::{
            zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1,
            zwp_relative_pointer_v1::{self, ZwpRelativePointerV1},
        },
    },
    xdg::{
        decoration::zv1::client::{
            zxdg_decoration_manager_v1::ZxdgDecorationManagerV1,
            zxdg_toplevel_decoration_v1::{self, ZxdgToplevelDecorationV1},
        },
        shell::client::{
            xdg_surface::{self, XdgSurface},
            xdg_toplevel::{self, XdgToplevel},
            xdg_wm_base::{self, XdgWmBase},
        },
    },
};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1},
};

use crate::{
    input::InputBackendId,
    shell::{
        CosmicMapped,
        focus::target::{KeyboardFocusTarget, PointerFocusTarget},
    },
    state::{ClientState, State},
    utils::prelude::*,
    wayland::handlers::compositor::client_compositor_state,
};

// ============================================================================
// Mock Input Backend for State::process_input_event
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MockDevice {
    id: String,
}

impl MockDevice {
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl Device for MockDevice {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn name(&self) -> String {
        format!("Mock Pointer {}", self.id)
    }
    fn has_capability(&self, capability: DeviceCapability) -> bool {
        matches!(
            capability,
            DeviceCapability::Pointer | DeviceCapability::Keyboard
        )
    }
    fn usb_id(&self) -> Option<(u32, u32)> {
        None
    }
    fn syspath(&self) -> Option<std::path::PathBuf> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct MockMotionEvent {
    pub device: MockDevice,
    pub time: u32,
    pub delta: Point<f64, Logical>,
    pub delta_unaccel: Point<f64, Logical>,
}

impl InputBackendEvent<MockInputBackend> for MockMotionEvent {
    fn time(&self) -> InputTime {
        InputTime::from_millis(self.time)
    }
    fn device(&self) -> MockDevice {
        self.device.clone()
    }
}

impl PointerMotionEvent<MockInputBackend> for MockMotionEvent {
    fn delta_x(&self) -> f64 {
        self.delta.x
    }
    fn delta_y(&self) -> f64 {
        self.delta.y
    }
    fn delta_x_unaccel(&self) -> f64 {
        self.delta_unaccel.x
    }
    fn delta_y_unaccel(&self) -> f64 {
        self.delta_unaccel.y
    }
}

#[derive(Clone, Debug)]
pub struct MockAbsoluteEvent {
    pub device: MockDevice,
    pub time: u32,
    pub norm_x: f64,
    pub norm_y: f64,
}

impl InputBackendEvent<MockInputBackend> for MockAbsoluteEvent {
    fn time(&self) -> InputTime {
        InputTime::from_millis(self.time)
    }
    fn device(&self) -> MockDevice {
        self.device.clone()
    }
}

impl AbsolutePositionEvent<MockInputBackend> for MockAbsoluteEvent {
    fn x(&self) -> f64 {
        self.norm_x
    }
    fn y(&self) -> f64 {
        self.norm_y
    }
    fn x_transformed(&self, width: i32) -> f64 {
        if width == 0 {
            self.norm_x
        } else {
            self.norm_x * (width as f64)
        }
    }
    fn y_transformed(&self, height: i32) -> f64 {
        if height == 0 {
            self.norm_y
        } else {
            self.norm_y * (height as f64)
        }
    }
}

impl PointerMotionAbsoluteEvent<MockInputBackend> for MockAbsoluteEvent {}

fn dummy_ei_backend_id() -> InputBackendId {
    let (sock, _peer) = UnixStream::pair().unwrap();
    let context = smithay::reexports::reis::eis::Context::new(sock).unwrap();
    InputBackendId::Ei(context.handshake().connection(1, 1))
}

#[derive(Clone, Debug)]
pub struct MockButtonEvent {
    pub device: MockDevice,
    pub time: u32,
    pub button_code: u32,
    pub state: ButtonState,
}

impl InputBackendEvent<MockInputBackend> for MockButtonEvent {
    fn time(&self) -> InputTime {
        InputTime::from_millis(self.time)
    }
    fn device(&self) -> MockDevice {
        self.device.clone()
    }
}

impl PointerButtonEvent<MockInputBackend> for MockButtonEvent {
    fn button_code(&self) -> u32 {
        self.button_code
    }
    fn state(&self) -> ButtonState {
        self.state
    }
}

pub struct MockInputBackend;

impl InputBackend for MockInputBackend {
    type Device = MockDevice;
    type KeyboardKeyEvent = UnusedEvent;
    type PointerAxisEvent = UnusedEvent;
    type PointerButtonEvent = MockButtonEvent;
    type PointerMotionEvent = MockMotionEvent;
    type PointerMotionAbsoluteEvent = MockAbsoluteEvent;
    type GestureSwipeBeginEvent = UnusedEvent;
    type GestureSwipeUpdateEvent = UnusedEvent;
    type GestureSwipeEndEvent = UnusedEvent;
    type GesturePinchBeginEvent = UnusedEvent;
    type GesturePinchUpdateEvent = UnusedEvent;
    type GesturePinchEndEvent = UnusedEvent;
    type GestureHoldBeginEvent = UnusedEvent;
    type GestureHoldEndEvent = UnusedEvent;
    type TouchDownEvent = UnusedEvent;
    type TouchUpEvent = UnusedEvent;
    type TouchMotionEvent = UnusedEvent;
    type TouchCancelEvent = UnusedEvent;
    type TouchFrameEvent = UnusedEvent;
    type TabletToolAxisEvent = UnusedEvent;
    type TabletToolProximityEvent = UnusedEvent;
    type TabletToolTipEvent = UnusedEvent;
    type TabletToolButtonEvent = UnusedEvent;
    type SwitchToggleEvent = UnusedEvent;
    type SpecialEvent = UnusedEvent;
}

// ============================================================================
// Wayland Test Client Infrastructure
// ============================================================================

#[derive(Default, Debug)]
pub struct ClientProtocolState {
    pub compositor: Option<WlCompositor>,
    pub subcompositor: Option<WlSubcompositor>,
    pub shm: Option<WlShm>,
    pub seat: Option<WlSeat>,
    pub pointer: Option<WlPointer>,
    pub wm_base: Option<XdgWmBase>,
    pub decoration_manager: Option<ZxdgDecorationManagerV1>,
    pub layer_shell: Option<ZwlrLayerShellV1>,
    pub relative_pointer_manager: Option<ZwpRelativePointerManagerV1>,
    pub relative_pointer: Option<ZwpRelativePointerV1>,
    pub pointer_constraints: Option<ZwpPointerConstraintsV1>,
    pub pointer_warp: Option<WpPointerWarpV1>,

    pub sync_done: bool,
    pub last_xdg_configure: Option<u32>,
    pub last_layer_configure: Option<(u32, u32, u32)>,

    pub pointer_enters: Vec<(ObjectId, f64, f64, u32)>,
    pub pointer_leaves: Vec<(ObjectId, u32)>,
    pub pointer_motions: Vec<(f64, f64)>,
    pub pointer_buttons: Vec<(u32, u32, WEnum<wl_pointer::ButtonState>)>,
    pub relative_motions: Vec<(f64, f64, f64, f64)>,

    pub locked_count: usize,
    pub unlocked_count: usize,
    pub confined_count: usize,
    pub unconfined_count: usize,
}

impl ClientProtocolState {
    pub fn clear_events(&mut self) {
        self.pointer_enters.clear();
        self.pointer_leaves.clear();
        self.pointer_motions.clear();
        self.pointer_buttons.clear();
        self.relative_motions.clear();
    }

    pub fn last_pointer_pos(&self) -> Option<(f64, f64)> {
        self.pointer_motions
            .last()
            .copied()
            .or_else(|| self.pointer_enters.last().map(|(_, x, y, _)| (*x, *y)))
    }

    pub fn last_enter_serial(&self) -> Option<u32> {
        self.pointer_enters.last().map(|(_, _, _, s)| *s)
    }
}

impl Dispatch<WlRegistry, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &ClientConnection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(6), qh, ()));
                }
                "wl_subcompositor" => {
                    state.subcompositor = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wl_seat" => {
                    let seat: WlSeat = registry.bind(name, version.min(8), qh, ());
                    let pointer = seat.get_pointer(qh, ());
                    if let Some(rel_mgr) = state.relative_pointer_manager.as_ref() {
                        state.relative_pointer =
                            Some(rel_mgr.get_relative_pointer(&pointer, qh, ()));
                    }
                    state.pointer = Some(pointer);
                    state.seat = Some(seat);
                }
                "xdg_wm_base" => {
                    state.wm_base = Some(registry.bind(name, version.min(5), qh, ()));
                }
                "zxdg_decoration_manager_v1" => {
                    state.decoration_manager = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(registry.bind(name, version.min(4), qh, ()));
                }
                "zwp_relative_pointer_manager_v1" => {
                    let mgr: ZwpRelativePointerManagerV1 =
                        registry.bind(name, version.min(1), qh, ());
                    if let Some(pointer) = state.pointer.as_ref() {
                        state.relative_pointer = Some(mgr.get_relative_pointer(pointer, qh, ()));
                    }
                    state.relative_pointer_manager = Some(mgr);
                }
                "zwp_pointer_constraints_v1" => {
                    state.pointer_constraints = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wp_pointer_warp_v1" => {
                    state.pointer_warp = Some(registry.bind(name, version.min(1), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<WlCallback, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        _: &WlCallback,
        event: wayland_client::protocol::wl_callback::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_callback::Event::Done { .. } = event {
            state.sync_done = true;
        }
    }
}

impl Dispatch<WlCompositor, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlCompositor,
        _: <WlCompositor as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSubcompositor, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlSubcompositor,
        _: <WlSubcompositor as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSubsurface, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlSubsurface,
        _: <WlSubsurface as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ClientWlSurface, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ClientWlSurface,
        _: <ClientWlSurface as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlRegion, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlRegion,
        _: <WlRegion as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShm, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlShm,
        _: wl_shm::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShmPool, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlBuffer, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlBuffer,
        _: <WlBuffer as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlSeat, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WlSeat,
        _: wl_seat::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlPointer, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        _: &WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                serial,
                surface,
                surface_x,
                surface_y,
            } => {
                state
                    .pointer_enters
                    .push((surface.id(), surface_x, surface_y, serial));
            }
            wl_pointer::Event::Leave { serial, surface } => {
                state.pointer_leaves.push((surface.id(), serial));
            }
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                state.pointer_motions.push((surface_x, surface_y));
            }
            wl_pointer::Event::Button {
                serial,
                button,
                state: btn_state,
                ..
            } => {
                state.pointer_buttons.push((serial, button, btn_state));
            }
            _ => {}
        }
    }
}

impl Dispatch<XdgWmBase, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        wm_base: &XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        xdg_surface: &XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            state.last_xdg_configure = Some(serial);
            xdg_surface.ack_configure(serial);
        }
    }
}

impl Dispatch<XdgToplevel, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &XdgToplevel,
        _: xdg_toplevel::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZxdgDecorationManagerV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ZxdgDecorationManagerV1,
        _: <ZxdgDecorationManagerV1 as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZxdgToplevelDecorationV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ZxdgToplevelDecorationV1,
        _: zxdg_toplevel_decoration_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ZwlrLayerShellV1,
        _: <ZwlrLayerShellV1 as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        layer_surface: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_layer_surface_v1::Event::Configure {
            serial,
            width,
            height,
        } = event
        {
            state.last_layer_configure = Some((serial, width, height));
            layer_surface.ack_configure(serial);
        }
    }
}

impl Dispatch<ZwpRelativePointerManagerV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ZwpRelativePointerManagerV1,
        _: <ZwpRelativePointerManagerV1 as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpRelativePointerV1, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        _: &ZwpRelativePointerV1,
        event: zwp_relative_pointer_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_relative_pointer_v1::Event::RelativeMotion {
            dx,
            dy,
            dx_unaccel,
            dy_unaccel,
            ..
        } = event
        {
            state
                .relative_motions
                .push((dx, dy, dx_unaccel, dy_unaccel));
        }
    }
}

impl Dispatch<ZwpPointerConstraintsV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &ZwpPointerConstraintsV1,
        _: <ZwpPointerConstraintsV1 as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwpLockedPointerV1, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        _: &ZwpLockedPointerV1,
        event: zwp_locked_pointer_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_locked_pointer_v1::Event::Locked => state.locked_count += 1,
            zwp_locked_pointer_v1::Event::Unlocked => state.unlocked_count += 1,
            _ => {}
        }
    }
}

impl Dispatch<ZwpConfinedPointerV1, ()> for ClientProtocolState {
    fn event(
        state: &mut Self,
        _: &ZwpConfinedPointerV1,
        event: zwp_confined_pointer_v1::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_confined_pointer_v1::Event::Confined => state.confined_count += 1,
            zwp_confined_pointer_v1::Event::Unconfined => state.unconfined_count += 1,
            _ => {}
        }
    }
}

impl Dispatch<WpPointerWarpV1, ()> for ClientProtocolState {
    fn event(
        _: &mut Self,
        _: &WpPointerWarpV1,
        _: <WpPointerWarpV1 as Proxy>::Event,
        _: &(),
        _: &ClientConnection,
        _: &QueueHandle<Self>,
    ) {
    }
}

pub struct WaylandTestClient {
    pub conn: ClientConnection,
    pub event_queue: EventQueue<ClientProtocolState>,
    pub qh: QueueHandle<ClientProtocolState>,
    pub state: ClientProtocolState,
    pub server_client: Client,
    _shm_fds: Vec<OwnedFd>,
}

#[allow(dead_code)]
pub struct TestWaylandWindow {
    pub client_surface: ClientWlSurface,
    pub xdg_surface: XdgSurface,
    pub xdg_toplevel: XdgToplevel,
    pub decoration: Option<ZxdgToplevelDecorationV1>,
    pub server_surface: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    pub mapped: CosmicMapped,
}

#[allow(dead_code)]
pub struct TestLayerSurface {
    pub client_surface: ClientWlSurface,
    pub layer_surface: ZwlrLayerSurfaceV1,
}

// ============================================================================
// TestCompositor Harness
// ============================================================================

pub struct TestCompositor {
    pub event_loop: EventLoop<'static, State>,
    pub display: Display<State>,
    pub state: State,
    pub device: MockDevice,
    pub time_counter: u32,
}

impl TestCompositor {
    pub fn new(outputs_spec: &[(&str, (i32, i32), f64, (i32, i32))]) -> Self {
        let _ = crate::hooks::HOOKS.set(crate::hooks::Hooks::default());
        let event_loop = EventLoop::<State>::try_new().expect("Failed to create EventLoop");
        let mut display = Display::<State>::new().expect("Failed to create Display");
        let dh = display.handle();

        // In production cosmic-comp, RelativePointerManagerState is initialized in kms::init_libinput
        // (src/backend/kms/mod.rs:398) rather than State::new.
        smithay::wayland::relative_pointer::RelativePointerManagerState::new::<State>(&dh);

        let socket_name = std::ffi::OsString::from(format!(
            "cosmic-test-{}-{}",
            std::process::id(),
            u32::from(SERIAL_COUNTER.next_serial())
        ));

        let mut state = State::new(
            &dh,
            socket_name,
            event_loop.handle(),
            event_loop.get_signal(),
            false,
            None,
        );

        let mut created_outputs = Vec::new();
        for (idx, (name, (w, h), scale, (x, y))) in outputs_spec.iter().enumerate() {
            let output = Output::new(
                (*name).to_string(),
                PhysicalProperties {
                    size: (300, 200).into(),
                    subpixel: Subpixel::HorizontalRgb,
                    make: "COSMIC".into(),
                    model: (*name).into(),
                    serial_number: format!("SN-{}", idx),
                },
            );
            let mode = OutputMode {
                size: (*w, *h).into(),
                refresh: 60_000,
            };
            output.add_mode(mode);
            output.set_preferred(mode);
            output.change_current_state(
                Some(mode),
                Some(Transform::Normal),
                Some(Scale::Fractional(*scale)),
                Some((*x, *y).into()),
            );
            output.user_data().insert_if_missing(|| {
                RefCell::new(OutputConfig {
                    mode: ((*w, *h), Some(60_000)),
                    vrr: AdaptiveSync::Disabled,
                    scale: *scale,
                    transform: TransformDef::Normal,
                    position: (*x as u32, *y as u32),
                    enabled: OutputState::Enabled,
                    xwayland_primary: idx == 0,
                    ..Default::default()
                })
            });
            created_outputs.push(output);
        }

        state
            .common
            .output_configuration_state
            .add_heads(created_outputs.iter());

        for output in &created_outputs {
            state.common.add_output(output);
            // Disable tiling by default so floating window tests have deterministic geometries
            let mut shell = state.common.shell.write();
            if let Some(ws) = shell.workspaces.active_mut(output) {
                ws.tiling_enabled = false;
            }
        }
        state.common.refresh();

        let first_output = created_outputs[0].clone();
        let initial_seat = crate::shell::create_seat(
            &dh,
            &mut state.common.seat_state,
            &first_output,
            &state.common.config,
            "seat-0".into(),
        );
        state.common.shell.write().seats.add_seat(initial_seat);

        // Register a Generic FD source on the display so client requests wake the event loop
        let display_fd = display.backend().poll_fd().try_clone_to_owned().unwrap();
        event_loop
            .handle()
            .insert_source(
                smithay::reexports::calloop::generic::Generic::new(
                    display_fd,
                    smithay::reexports::calloop::Interest::READ,
                    smithay::reexports::calloop::Mode::Level,
                ),
                |_, _, _| Ok(smithay::reexports::calloop::PostAction::Continue),
            )
            .unwrap();

        let device = MockDevice::new("mock-mouse-0");
        state.process_input_event(
            InputEvent::<MockInputBackend>::DeviceAdded {
                device: device.clone(),
            },
            InputBackendId::Normal,
        );

        state.common.update_xwayland_primary_output();
        state.common.update_xwayland_settings();

        Self {
            event_loop,
            display,
            state,
            device,
            time_counter: 100,
        }
    }

    pub fn dispatch(&mut self) {
        self.display.dispatch_clients(&mut self.state).unwrap();
        self.event_loop
            .dispatch(Some(Duration::from_millis(1)), &mut self.state)
            .unwrap();
        self.display.dispatch_clients(&mut self.state).unwrap();
        self.display.flush_clients().unwrap();
    }

    pub fn new_wayland_client(&mut self) -> WaylandTestClient {
        let (server_stream, client_stream) = UnixStream::pair().unwrap();
        server_stream.set_nonblocking(true).unwrap();
        client_stream.set_nonblocking(true).unwrap();

        let client_state = Arc::new(ClientState {
            compositor_client_state: Default::default(),
            advertised_drm_node: None,
            evlh: self.state.common.event_loop_handle.clone(),
            evls: self.state.common.event_loop_signal.clone(),
            security_context: None,
        });
        let server_client = self
            .display
            .handle()
            .insert_client(server_stream, client_state)
            .unwrap();

        let conn = ClientConnection::from_socket(client_stream).unwrap();
        let event_queue = conn.new_event_queue::<ClientProtocolState>();
        let qh = event_queue.handle();
        let display_proxy = conn.display();
        let _registry = display_proxy.get_registry(&qh, ());

        let mut client = WaylandTestClient {
            conn,
            event_queue,
            qh,
            state: ClientProtocolState::default(),
            server_client,
            _shm_fds: Vec::new(),
        };

        // Two roundtrips so registry globals AND nested binds (wl_pointer, relative_pointer) complete
        self.roundtrip(&mut client);
        self.roundtrip(&mut client);
        client
    }

    pub fn roundtrip(&mut self, client: &mut WaylandTestClient) {
        client.state.sync_done = false;
        let _cb = client.conn.display().sync(&client.qh, ());
        client.conn.flush().unwrap();

        for _ in 0..50 {
            self.dispatch();
            let _ = client.event_queue.dispatch_pending(&mut client.state);
            let _ = client.conn.flush();
            if let Some(guard) = client.event_queue.prepare_read() {
                let _ = guard.read();
            }
            let _ = client.event_queue.dispatch_pending(&mut client.state);
            if client.state.sync_done {
                break;
            }
        }
        assert!(client.state.sync_done, "Wayland client roundtrip timed out");
    }

    pub fn flush_to_client(&mut self, client: &mut WaylandTestClient) {
        self.roundtrip(client);
    }

    pub fn create_shm_buffer(
        &mut self,
        client: &mut WaylandTestClient,
        width: i32,
        height: i32,
    ) -> WlBuffer {
        let stride = width * 4;
        let size = stride * height;
        let fd = rustix::fs::memfd_create("cosmic-test-shm", rustix::fs::MemfdFlags::CLOEXEC)
            .expect("Failed to create memfd");
        rustix::fs::ftruncate(&fd, size as u64).expect("Failed to ftruncate memfd");

        let shm = client.state.shm.as_ref().expect("wl_shm not bound");
        let pool = shm.create_pool(fd.as_fd(), size, &client.qh, ());
        let buffer = pool.create_buffer(
            0,
            width,
            height,
            stride,
            wl_shm::Format::Argb8888,
            &client.qh,
            (),
        );
        pool.destroy();
        client._shm_fds.push(fd);
        buffer
    }

    pub fn map_xdg_window(
        &mut self,
        client: &mut WaylandTestClient,
        pos: Point<i32, Local>,
        size: Size<i32, Logical>,
        ssd: bool,
        window_geometry: Option<Rectangle<i32, Logical>>,
    ) -> TestWaylandWindow {
        let compositor = client.state.compositor.as_ref().unwrap();
        let wm_base = client.state.wm_base.as_ref().unwrap();

        let client_surface = compositor.create_surface(&client.qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&client_surface, &client.qh, ());
        let xdg_toplevel = xdg_surface.get_toplevel(&client.qh, ());
        xdg_toplevel.set_title("TestWindow".into());
        xdg_toplevel.set_app_id("com.system76.TestWindow".into());

        let decoration = if let Some(dec_mgr) = client.state.decoration_manager.as_ref() {
            let dec = dec_mgr.get_toplevel_decoration(&xdg_toplevel, &client.qh, ());
            dec.set_mode(if ssd {
                zxdg_toplevel_decoration_v1::Mode::ServerSide
            } else {
                zxdg_toplevel_decoration_v1::Mode::ClientSide
            });
            Some(dec)
        } else {
            None
        };

        client_surface.commit();
        self.roundtrip(client);

        if let Some(geo) = window_geometry {
            xdg_surface.set_window_geometry(geo.loc.x, geo.loc.y, geo.size.w, geo.size.h);
        }

        let buffer = self.create_shm_buffer(client, size.w, size.h);
        client_surface.attach(Some(&buffer), 0, 0);
        client_surface.damage_buffer(0, 0, size.w, size.h);
        client_surface.commit();
        self.roundtrip(client);

        let mapped = self
            .state
            .common
            .shell
            .read()
            .mapped()
            .find(|m| {
                m.active_window()
                    .wl_surface()
                    .is_some_and(|s| s.id().protocol_id() == client_surface.id().protocol_id())
            })
            .cloned()
            .expect("Window should be mapped in Shell");

        let server_surface = mapped.active_window().wl_surface().unwrap().into_owned();

        // Place at exact local position on active workspace without open/tile animation
        {
            let mut shell = self.state.common.shell.write();
            let seat = shell.seats.last_active().clone();
            let output = seat.active_output();
            let ws = shell.workspaces.active_mut(&output).unwrap();
            mapped.set_geometry(Rectangle::new(pos, size.as_local()).to_global(&output));
            ws.floating_layer
                .space
                .map_element(mapped.clone(), pos.as_logical(), false);
            ws.floating_layer.space.refresh();
        }
        self.roundtrip(client);

        TestWaylandWindow {
            client_surface,
            xdg_surface,
            xdg_toplevel,
            decoration,
            server_surface,
            mapped,
        }
    }

    pub fn map_layer_surface(
        &mut self,
        client: &mut WaylandTestClient,
        layer: Layer,
        size: (u32, u32),
        anchor: Anchor,
    ) -> TestLayerSurface {
        let compositor = client.state.compositor.as_ref().unwrap();
        let layer_shell = client.state.layer_shell.as_ref().unwrap();

        let client_surface = compositor.create_surface(&client.qh, ());
        let layer_surface = layer_shell.get_layer_surface(
            &client_surface,
            None,
            layer,
            "test-layer".into(),
            &client.qh,
            (),
        );
        layer_surface.set_size(size.0, size.1);
        layer_surface.set_anchor(anchor);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::OnDemand);
        client_surface.commit();
        self.roundtrip(client);

        let buffer = self.create_shm_buffer(client, size.0 as i32, size.1 as i32);
        client_surface.attach(Some(&buffer), 0, 0);
        client_surface.damage_buffer(0, 0, size.0 as i32, size.1 as i32);
        client_surface.commit();
        self.roundtrip(client);

        let outputs = self.outputs();
        for o in &outputs {
            layer_map_for_output(o).arrange();
        }

        TestLayerSurface {
            client_surface,
            layer_surface,
        }
    }

    pub fn inject_motion(
        &mut self,
        dx: f64,
        dy: f64,
        dx_unaccel: f64,
        dy_unaccel: f64,
        backend_id: InputBackendId,
    ) {
        self.time_counter += 8;
        let event = InputEvent::<MockInputBackend>::PointerMotion {
            event: MockMotionEvent {
                device: self.device.clone(),
                time: self.time_counter,
                delta: (dx, dy).into(),
                delta_unaccel: (dx_unaccel, dy_unaccel).into(),
            },
        };
        self.state.process_input_event(event, backend_id);
    }

    pub fn inject_motion_absolute(
        &mut self,
        norm_x: f64,
        norm_y: f64,
        backend_id: InputBackendId,
    ) {
        self.time_counter += 8;
        let event = InputEvent::<MockInputBackend>::PointerMotionAbsolute {
            event: MockAbsoluteEvent {
                device: self.device.clone(),
                time: self.time_counter,
                norm_x,
                norm_y,
            },
        };
        self.state.process_input_event(event, backend_id);
    }

    pub fn inject_button(
        &mut self,
        button_code: u32,
        state: ButtonState,
        backend_id: InputBackendId,
    ) {
        self.time_counter += 8;
        let event = InputEvent::<MockInputBackend>::PointerButton {
            event: MockButtonEvent {
                device: self.device.clone(),
                time: self.time_counter,
                button_code,
                state,
            },
        };
        self.state.process_input_event(event, backend_id);
    }

    pub fn pointer_global_location(&self) -> Point<f64, Global> {
        let shell = self.state.common.shell.read();
        let seat = shell.seats.last_active();
        seat.get_pointer().unwrap().current_location().as_global()
    }

    pub fn active_output(&self) -> Output {
        let shell = self.state.common.shell.read();
        shell.seats.last_active().active_output()
    }

    pub fn outputs(&self) -> Vec<Output> {
        self.state
            .common
            .shell
            .read()
            .outputs()
            .cloned()
            .collect()
    }

    pub fn surface_under(
        &self,
        global_pos: Point<f64, Global>,
        output: &Output,
    ) -> Option<(PointerFocusTarget, Point<f64, Global>)> {
        let shell = self.state.common.shell.read();
        State::surface_under(global_pos, output, &shell)
    }
}

// ============================================================================
// 1. Resolution, Scaling, Relative & Absolute Motion Matrix Tests
// ============================================================================

#[test]
fn test_relative_and_absolute_motion_across_resolutions_and_scales() {
    let configs = [
        ("FHD-100", (1920, 1080), 1.0),
        ("QHD-125", (2560, 1440), 1.25),
        ("Retina-150", (2880, 1800), 1.5),
        ("UHD-175", (3840, 2160), 1.75),
        ("UHD-200", (3840, 2160), 2.0),
    ];

    for (name, (phys_w, phys_h), scale) in configs {
        let mut comp = TestCompositor::new(&[(name, (phys_w, phys_h), scale, (0, 0))]);
        let mut client = comp.new_wayland_client();

        let output = comp.active_output();
        let out_geo = output.geometry();
        let expected_log_w = (phys_w as f64 / scale).round() as i32;
        let expected_log_h = (phys_h as f64 / scale).round() as i32;
        assert_eq!(
            out_geo.size,
            Size::from((expected_log_w, expected_log_h)),
            "Output logical size mismatch for {name}"
        );

        let win = comp.map_xdg_window(
            &mut client,
            Point::from((100, 100)),
            Size::from((400, 300)),
            false,
            None,
        );

        client.state.clear_events();

        // 1. Absolute motion (Normal backend) into window at logical (250, 220)
        let norm_x = 250.0 / (out_geo.size.w as f64);
        let norm_y = 220.0 / (out_geo.size.h as f64);
        comp.inject_motion_absolute(norm_x, norm_y, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 250.0).abs() < 1e-3);
        assert!((pos.y - 220.0).abs() < 1e-3);
        assert_eq!(client.state.pointer_enters.len(), 1);
        assert_eq!(client.state.pointer_enters[0].0, win.client_surface.id());
        assert!((client.state.pointer_enters[0].1 - 150.0).abs() < 1e-2);
        assert!((client.state.pointer_enters[0].2 - 120.0).abs() < 1e-2);

        client.state.clear_events();

        // 2. Relative motion (Normal backend) inside window: dx=15.5, dy=-10.25, unaccel=(12.0, -8.0)
        comp.inject_motion(15.5, -10.25, 12.0, -8.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 265.5).abs() < 1e-3);
        assert!((pos.y - 209.75).abs() < 1e-3);
        assert_eq!(client.state.pointer_motions.len(), 1);
        assert!((client.state.pointer_motions[0].0 - 165.5).abs() < 1e-2);
        assert!((client.state.pointer_motions[0].1 - 109.75).abs() < 1e-2);
        assert_eq!(client.state.relative_motions.len(), 1);
        assert_eq!(
            client.state.relative_motions[0],
            (15.5, -10.25, 12.0, -8.0)
        );

        client.state.clear_events();

        // 3. Relative & Absolute motion via Libei backend (InputBackendId::Ei)
        comp.inject_motion(-5.5, 0.25, -5.5, 0.25, dummy_ei_backend_id());
        comp.flush_to_client(&mut client);
        assert!((comp.pointer_global_location().x - 260.0).abs() < 1e-3);
        assert!((comp.pointer_global_location().y - 210.0).abs() < 1e-3);

        // EI absolute coordinates are in global logical space
        comp.inject_motion_absolute(300.0, 250.0, dummy_ei_backend_id());
        comp.flush_to_client(&mut client);
        assert!((comp.pointer_global_location().x - 300.0).abs() < 1e-3);
        assert!((comp.pointer_global_location().y - 250.0).abs() < 1e-3);

        // 4. Right/Bottom boundary clamping comparison:
        // Relative motion clamps to `(out_geo.size.w as f64).next_down()` (strictly inside Output::geometry().contains)
        comp.inject_motion(100_000.0, 100_000.0, 100_000.0, 100_000.0, InputBackendId::Normal);
        let rel_clamped = comp.pointer_global_location();
        assert_eq!(rel_clamped.x, (out_geo.size.w as f64).next_down());
        assert_eq!(rel_clamped.y, (out_geo.size.h as f64).next_down());
        assert!(out_geo.to_f64().contains(rel_clamped));

        // Finding I: PointerMotionAbsolute at norm=(1.0, 1.0) sets position to exact (w, h),
        // which is outside half-open Rectangle::contains [0, w) x [0, h)!
        comp.inject_motion_absolute(1.0, 1.0, InputBackendId::Normal);
        let abs_edge = comp.pointer_global_location();
        assert_eq!(abs_edge.x, out_geo.size.w as f64);
        assert_eq!(abs_edge.y, out_geo.size.h as f64);
        assert!(
            !out_geo.to_f64().contains(abs_edge),
            "Finding I verified: PointerMotionAbsolute(1.0, 1.0) lands on exact exclusive upper bound"
        );
    }
}

// ============================================================================
// 2. Finding E: PointerMotion Old `under` 1-Frame Entry/Leave Lag
// ============================================================================

#[test]
fn test_pointer_motion_old_under_one_frame_lag_on_window_entry() {
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();

    let win = comp.map_xdg_window(
        &mut client,
        Point::from((100, 100)),
        Size::from((400, 300)),
        false,
        None,
    );

    // Place pointer just outside the left edge of the window at (90, 200)
    comp.inject_motion_absolute(90.0 / 1920.0, 200.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    client.state.clear_events();

    // Move pointer by +20px in X to (110, 200), which is inside the window [100..500, 100..400].
    comp.inject_motion(20.0, 0.0, 20.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    assert_eq!(comp.pointer_global_location(), Point::from((110.0, 200.0)));
    // Finding E: Because src/input/mod.rs:366 computes `under` at the OLD position (90, 200)
    // and passes `under` (None) instead of `new_under` to `ptr.motion(...)` at line 646,
    // the client does NOT receive `wl_pointer.enter` on the frame the cursor enters the window!
    assert!(
        client.state.pointer_enters.is_empty(),
        "Finding E verified: wl_pointer.enter is delayed by 1 motion event because old `under` is passed to ptr.motion"
    );

    // On the NEXT motion event (+5px to (115, 200)), old position is (110, 200) which is inside the window,
    // so `wl_pointer.enter` finally arrives!
    comp.inject_motion(5.0, 0.0, 5.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(client.state.pointer_enters.len(), 1);
    assert_eq!(client.state.pointer_enters[0].0, win.client_surface.id());
    assert!((client.state.pointer_enters[0].1 - 15.0).abs() < 1e-4);
}

// ============================================================================
// 3. Pointer Constraints: Lock, Unlock, Oneshot vs Persistent, SSD Bypass
// ============================================================================

#[test]
fn test_pointer_lock_unlock_oneshot_persistent_and_ssd_bypass() {
    for scale in [1.0, 1.5, 2.0] {
        let mut comp = TestCompositor::new(&[("DP-1", (3840, 2160), scale, (0, 0))]);
        let mut client = comp.new_wayland_client();
        let out_size = comp.active_output().geometry().size;

        let win = comp.map_xdg_window(
            &mut client,
            Point::from((100, 100)),
            Size::from((400, 300)),
            false,
            None,
        );

        // Move pointer inside window to (250, 250)
        comp.inject_motion_absolute(
            250.0 / (out_size.w as f64),
            250.0 / (out_size.h as f64),
            InputBackendId::Normal,
        );
        comp.flush_to_client(&mut client);
        assert!((comp.pointer_global_location().x - 250.0).abs() < 1e-6);
        assert!((comp.pointer_global_location().y - 250.0).abs() < 1e-6);

        // 1. Lock pointer with Lifetime::Oneshot
        let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
        let pointer = client.state.pointer.as_ref().unwrap().clone();
        let locked_ptr = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Oneshot,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        assert_eq!(client.state.locked_count, 1);

        client.state.clear_events();

        // Inject relative motion while locked: global position must remain frozen at (250, 250),
        // wl_pointer.motion must NOT be sent, and relative_pointer must receive exact deltas.
        comp.inject_motion(42.5, -18.25, 40.0, -16.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        assert!((comp.pointer_global_location().x - 250.0).abs() < 1e-6);
        assert!((comp.pointer_global_location().y - 250.0).abs() < 1e-6);
        assert!(client.state.pointer_motions.is_empty());
        assert_eq!(
            client.state.relative_motions,
            vec![(42.5, -18.25, 40.0, -16.0)]
        );

        // Destroy locked_pointer to unlock
        locked_ptr.destroy();
        comp.roundtrip(&mut client);

        // Now relative motion moves the global pointer again
        comp.inject_motion(10.0, 5.0, 10.0, 5.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);
        assert!((comp.pointer_global_location().x - 260.0).abs() < 1e-6);
        assert!((comp.pointer_global_location().y - 255.0).abs() < 1e-6);

        // 2. Lock pointer with Lifetime::Persistent and test unfocus/refocus reactivation
        let persistent_lock = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Persistent,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        assert_eq!(client.state.locked_count, 2);

        // Unfocus the window -> should deactivate constraint and send `unlocked`
        {
            let seat = comp.state.common.shell.read().seats.last_active().clone();
            crate::shell::Shell::set_focus(&mut comp.state, None, &seat, None, false);
        }
        comp.roundtrip(&mut client);
        assert_eq!(client.state.unlocked_count, 1);

        // Refocus the window and move pointer inside -> should reactivate Persistent lock!
        {
            let seat = comp.state.common.shell.read().seats.last_active().clone();
            let target = KeyboardFocusTarget::Element(win.mapped.clone());
            crate::shell::Shell::set_focus(&mut comp.state, Some(&target), &seat, None, false);
        }
        comp.inject_motion(1.0, 1.0, 1.0, 1.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);
        assert_eq!(client.state.locked_count, 3);

        persistent_lock.destroy();
        comp.roundtrip(&mut client);
    }

    // 3. Test SSD headerbar bypass (TC-LOCK-06):
    // When a window has ServerSide decorations, hovering over the SSD headerbar (above the client surface)
    // yields PointerFocusTarget::Window(CosmicWindow) instead of PointerFocusTarget::WlSurface,
    // which must NOT activate a surface lock constraint.
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();
    let ssd_win = comp.map_xdg_window(
        &mut client,
        Point::from((200, 200)),
        Size::from((400, 300)),
        true,
        None,
    );

    // Move pointer to (50, 50) outside window first
    comp.inject_motion_absolute(50.0 / 1920.0, 50.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();
    let _ssd_lock = constraints.lock_pointer(
        &ssd_win.client_surface,
        &pointer,
        None,
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    assert_eq!(client.state.locked_count, 0);

    // Move pointer onto the top SSD titlebar area at (300, 205)
    comp.inject_motion_absolute(300.0 / 1920.0, 205.0 / 1080.0, InputBackendId::Normal);
    comp.inject_motion(1.0, 0.0, 1.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(
        client.state.locked_count, 0,
        "Hovering SSD headerbar must not trigger client surface pointer lock"
    );

    // Move pointer into client content area at (300, 300) -> lock activates!
    comp.inject_motion_absolute(300.0 / 1920.0, 300.0 / 1080.0, InputBackendId::Normal);
    comp.inject_motion(1.0, 0.0, 1.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(client.state.locked_count, 1);
}

// ============================================================================
// 4. Cursor Position Hint & apply_cursor_hint + Finding B (CSD Offset)
// ============================================================================

#[test]
fn test_pointer_lock_cursor_hint_and_csd_geometry_offset() {
    for scale in [1.0, 1.25, 1.5, 2.0] {
        let mut comp = TestCompositor::new(&[("DP-1", (3840, 2160), scale, (0, 0))]);
        let mut client = comp.new_wayland_client();
        let out_size = comp.active_output().geometry().size;

        let win = comp.map_xdg_window(
            &mut client,
            Point::from((100, 100)),
            Size::from((400, 300)),
            false,
            None,
        );

        comp.inject_motion_absolute(
            200.0 / (out_size.w as f64),
            200.0 / (out_size.h as f64),
            InputBackendId::Normal,
        );
        comp.flush_to_client(&mut client);

        let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
        let pointer = client.state.pointer.as_ref().unwrap().clone();

        // 1. Valid cursor position hint (320, 240) inside 400x300 window -> warps to (100+320, 100+240) = (420, 340)
        let locked_ptr = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Oneshot,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        assert_eq!(client.state.locked_count, 1);

        locked_ptr.set_cursor_position_hint(320.0, 240.0);
        win.client_surface.commit();
        comp.roundtrip(&mut client);

        client.state.clear_events();
        locked_ptr.destroy();
        comp.roundtrip(&mut client);

        assert_eq!(comp.pointer_global_location(), Point::from((420.0, 340.0)));
        assert_eq!(client.state.last_pointer_pos(), Some((320.0, 240.0)));

        // 2. Out-of-bounds cursor position hint (999.0, 999.0) -> rejected by apply_cursor_hint, stays at (420, 340)
        let locked_ptr2 = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Oneshot,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        locked_ptr2.set_cursor_position_hint(999.0, 999.0);
        win.client_surface.commit();
        comp.roundtrip(&mut client);
        locked_ptr2.destroy();
        comp.roundtrip(&mut client);

        assert_eq!(
            comp.pointer_global_location(),
            Point::from((420.0, 340.0)),
            "Out-of-bounds cursor position hint must be ignored"
        );
    }

    // 3. Finding B: CSD window with non-zero xdg_surface.set_window_geometry(24, 24, 400, 300)
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();
    let csd_win = comp.map_xdg_window(
        &mut client,
        Point::from((100, 100)),
        Size::from((448, 348)),
        false,
        Some(Rectangle::new(Point::from((24, 24)), Size::from((400, 300)))),
    );

    // Move pointer to global (200, 200).
    // FloatingLayout::toplevel_surface_under subtracts window_geometry.loc (24, 24) from window origin (100, 100),
    // placing the wl_surface origin at (76, 76), so surface-local coords at global (200, 200) are (124, 124).
    comp.inject_motion_absolute(200.0 / 1920.0, 200.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(client.state.last_pointer_pos(), Some((124.0, 124.0)));

    // Now lock and set cursor position hint to (150, 150)
    let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();
    let locked_ptr = constraints.lock_pointer(
        &csd_win.client_surface,
        &pointer,
        None,
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    locked_ptr.set_cursor_position_hint(150.0, 150.0);
    csd_win.client_surface.commit();
    comp.roundtrip(&mut client);

    client.state.clear_events();
    locked_ptr.destroy();
    comp.roundtrip(&mut client);

    // Finding B: surface_geometry_offset_from_toplevel does NOT subtract window_geometry.loc (24, 24),
    // so apply_cursor_hint warps global pointer to (100 + 150, 100 + 150) = (250, 250),
    // and then State::surface_under subtracts (76, 76), reporting wl_pointer.motion(174.0, 174.0) instead of (150.0, 150.0)!
    assert_eq!(comp.pointer_global_location(), Point::from((250.0, 250.0)));
    assert_eq!(
        client.state.last_pointer_pos(),
        Some((174.0, 174.0)),
        "Finding B verified: CSD window_geometry.loc offset causes 24px discrepancy between hint and resulting wl_pointer.motion"
    );
}

// ============================================================================
// 5. Pointer Confinement, Custom wl_region, and Finding F (Axis Sliding Bug)
// ============================================================================

#[test]
fn test_pointer_confine_region_and_axis_sliding_bug() {
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();

    let win = comp.map_xdg_window(
        &mut client,
        Point::from((100, 100)),
        Size::from((400, 300)),
        false,
        None,
    );

    // Move pointer inside window to (250, 250) (surface-local 150, 150)
    comp.inject_motion_absolute(250.0 / 1920.0, 250.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    // Create a custom confinement wl_region: [50, 50, 200, 200] in surface-local coords
    // (which corresponds to global [150..350, 150..350])
    let region = client
        .state
        .compositor
        .as_ref()
        .unwrap()
        .create_region(&client.qh, ());
    region.add(50, 50, 200, 200);

    let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();
    let confined_ptr = constraints.confine_pointer(
        &win.client_surface,
        &pointer,
        Some(&region),
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    assert_eq!(client.state.confined_count, 1);

    // 1. Motion inside the confined region (+20, +30) -> allowed, moves to (270, 280)
    comp.inject_motion(20.0, 30.0, 20.0, 30.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(comp.pointer_global_location(), Point::from((270.0, 280.0)));

    // 2. Pure horizontal motion trying to exit right edge of region [150..350] (+200, 0):
    // Target is (470, 280), which is outside region (even if inside window).
    // Note: in src/input/mod.rs:607, `new_under` is still `Some((win, (100, 100)))`, so it checks `region.contains`.
    // And in `is_legal` (line 576), `position` is (470, 280), which is outside `region`, so both X and Y are rejected!
    comp.inject_motion(200.0, 0.0, 200.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((270.0, 280.0)),
        "Pointer must not escape custom confined wl_region"
    );

    // 3. Finding F: Diagonal motion (dx = +300, dy = +20) trying to slide along the right edge:
    // Old position is (270, 280).
    // `y_only_pos` is (270, 300), which is completely legal (inside region [150..350, 150..350] and window [100..500, 100..400]).
    // However, `is_legal` in src/input/mod.rs:576, 588 checks `position.as_logical() - s_pos` (570, 300)
    // instead of `pos.as_logical() - s_pos` (270, 300)!
    // As a result, `is_legal(y_only_pos)` returns false and blocks the vertical slide!
    comp.inject_motion(300.0, 20.0, 300.0, 20.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((270.0, 280.0)),
        "Finding F verified: Confined diagonal motion fails to slide along valid Y axis because is_legal checks `position` instead of `pos`"
    );

    confined_ptr.destroy();
    comp.roundtrip(&mut client);
}

// ============================================================================
// 6. wp_pointer_warp_v1 Protocol & Finding A (client_scale Double Downscale)
// ============================================================================

#[test]
fn test_pointer_warp_protocol_and_client_scale_double_downscale() {
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();

    let win = comp.map_xdg_window(
        &mut client,
        Point::from((100, 100)),
        Size::from((400, 300)),
        false,
        None,
    );

    let warp_mgr = client.state.pointer_warp.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();

    // 1. TC-WARP-03: Warp while pointer is outside the window (unfocused) -> must be ignored
    comp.inject_motion_absolute(50.0 / 1920.0, 50.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 200.0, 150.0, 1);
    comp.roundtrip(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((50.0, 50.0)),
        "Warp while pointer is not focused on client surface must be rejected"
    );

    // Move pointer into window at (150, 150) to obtain a valid enter serial
    comp.inject_motion_absolute(150.0 / 1920.0, 150.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    let enter_serial = client.state.last_enter_serial().expect("Expected enter serial");

    // 2. TC-WARP-04: Warp with invalid serial (999999) -> must be ignored
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 250.0, 200.0, 999_999);
    comp.roundtrip(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((150.0, 150.0)),
        "Warp with invalid serial must be rejected"
    );

    // 3. TC-WARP-05: Warp with valid serial but out-of-bounds coordinates (500.0, 500.0) -> must be ignored
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 500.0, 500.0, enter_serial);
    comp.roundtrip(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((150.0, 150.0)),
        "Out-of-bounds warp must be rejected"
    );

    // 4. TC-WARP-01: Valid warp on toplevel to (250.0, 200.0) -> global (350.0, 300.0)
    client.state.clear_events();
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 250.0, 200.0, enter_serial);
    comp.roundtrip(&mut client);
    assert_eq!(comp.pointer_global_location(), Point::from((350.0, 300.0)));
    assert_eq!(client.state.last_pointer_pos(), Some((250.0, 200.0)));

    // 5. TC-WARP-02: Valid warp targeting a child wl_subsurface at offset (50, 40)
    let subcompositor = client.state.subcompositor.as_ref().unwrap().clone();
    let compositor = client.state.compositor.as_ref().unwrap().clone();
    let child_surface = compositor.create_surface(&client.qh, ());
    let subsurface =
        subcompositor.get_subsurface(&child_surface, &win.client_surface, &client.qh, ());
    subsurface.set_position(50, 40);
    subsurface.set_desync();
    let child_buf = comp.create_shm_buffer(&mut client, 100, 100);
    child_surface.attach(Some(&child_buf), 0, 0);
    child_surface.damage_buffer(0, 0, 100, 100);
    child_surface.commit();
    win.client_surface.commit();
    comp.roundtrip(&mut client);

    // Warp to (30, 20) on child_surface -> window-local (50+30, 40+20) = (80, 60) -> global (180, 160)
    warp_mgr.warp_pointer(&child_surface, &pointer, 30.0, 20.0, enter_serial);
    comp.roundtrip(&mut client);
    assert_eq!(comp.pointer_global_location(), Point::from((180.0, 160.0)));

    // Move back to parent surface outside child_surface so parent receives a fresh enter_serial
    comp.inject_motion_absolute(350.0 / 1920.0, 300.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    let enter_serial = client
        .state
        .last_enter_serial()
        .expect("Expected updated enter serial on parent");

    // 6. Verify Anomaly A fix: both wp_pointer_warp_v1 and zwp_locked_pointer_v1.set_cursor_position_hint
    // divide by client_scale exactly once when client_scale != 1.0 (e.g. client_scale = 2.0)
    client_compositor_state(&client.server_client).set_client_scale(2.0);
    // Client requests warp to (200.0, 160.0) in client coordinates (which at client_scale=2.0 means (100.0, 80.0) logical).
    // Expected global position: (100 + 100, 100 + 80) = (200, 180).
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 200.0, 160.0, enter_serial);
    comp.roundtrip(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((200.0, 180.0)),
        "wp_pointer_warp_v1 should divide by client_scale=2.0 exactly once"
    );

    // Also verify zwp_locked_pointer_v1.set_cursor_position_hint under client_scale = 2.0
    let constraints = client
        .state
        .pointer_constraints
        .as_ref()
        .unwrap()
        .clone();
    let locked = constraints.lock_pointer(
        &win.client_surface,
        &pointer,
        None,
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    assert_eq!(client.state.locked_count, 1);

    // Client sets hint (240.0, 180.0) in client coordinates -> (120.0, 90.0) logical -> global (220.0, 190.0)
    locked.set_cursor_position_hint(240.0, 180.0);
    win.client_surface.commit();
    comp.roundtrip(&mut client);
    locked.destroy();
    comp.roundtrip(&mut client);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((220.0, 190.0)),
        "set_cursor_position_hint should divide by client_scale=2.0 exactly once in Smithay"
    );
}

// ============================================================================
// 7. Multi-Monitor Topologies: Horizontal, Vertical, L-Shaped Offset, Disjoint Gap
// ============================================================================

#[test]
fn test_multi_monitor_topologies_traversal_clamping_and_gap() {
    // 1. M7 & M9: L-Shaped Dual Monitor Setup
    // DP-1: 1920x1080 @ 1.0 at (0, 0) -> [0..1920, 0..1080]
    // DP-2: 2560x1440 @ 1.25 (logical 2048x1152) at (1920, 500) -> [1920..3968, 500..1652]
    let mut comp = TestCompositor::new(&[
        ("DP-1", (1920, 1080), 1.0, (0, 0)),
        ("DP-2", (2560, 1440), 1.25, (1920, 500)),
    ]);

    // Move to (1900, 200) on DP-1 (where DP-2 does NOT exist to the right because DP-2 starts at y=500)
    comp.inject_motion_absolute(1900.0 / 1920.0, 200.0 / 1080.0, InputBackendId::Normal);
    assert_eq!(comp.active_output().name(), "DP-1");

    // Push right by +100 and down by +50: X must clamp to DP-1's right edge (1920.0.next_down()), Y slides to 250.0
    comp.inject_motion(100.0, 50.0, 100.0, 50.0, InputBackendId::Normal);
    assert_eq!(comp.active_output().name(), "DP-1");
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((1920.0f64.next_down(), 250.0))
    );

    // Now move down to y=700 on DP-1 (where DP-2 IS adjacent at x=1920) and push right by +100
    comp.inject_motion(0.0, 450.0, 0.0, 450.0, InputBackendId::Normal);
    assert_eq!(
        comp.pointer_global_location(),
        Point::from((1920.0f64.next_down(), 700.0))
    );
    comp.inject_motion(100.0, 0.0, 100.0, 0.0, InputBackendId::Normal);
    assert_eq!(comp.active_output().name(), "DP-2");
    assert!((comp.pointer_global_location().x - 2020.0).abs() < 1e-3);
    assert_eq!(comp.pointer_global_location().y, 700.0);

    // 2. M8: Vertical Dual Monitor Setup
    // DP-1: 1920x1080 @ 1.0 at (0, 0)
    // DP-2: 1920x1080 @ 1.0 at (0, 1080)
    let mut vcomp = TestCompositor::new(&[
        ("DP-1", (1920, 1080), 1.0, (0, 0)),
        ("DP-2", (1920, 1080), 1.0, (0, 1080)),
    ]);
    vcomp.inject_motion_absolute(500.0 / 1920.0, 1050.0 / 1080.0, InputBackendId::Normal);
    assert_eq!(vcomp.active_output().name(), "DP-1");
    vcomp.inject_motion(0.0, 100.0, 0.0, 100.0, InputBackendId::Normal);
    assert_eq!(vcomp.active_output().name(), "DP-2");
    assert!((vcomp.pointer_global_location().x - 500.0).abs() < 1e-6);
    assert!((vcomp.pointer_global_location().y - 1150.0).abs() < 1e-6);

    // 3. M10: Disjoint Monitors with 100px Gap
    // DP-1: 1920x1080 @ 1.0 at (0, 0)
    // DP-2: 1920x1080 @ 1.0 at (2020, 0) (gap [1920..2020])
    let mut gcomp = TestCompositor::new(&[
        ("DP-1", (1920, 1080), 1.0, (0, 0)),
        ("DP-2", (1920, 1080), 1.0, (2020, 0)),
    ]);
    gcomp.inject_motion_absolute(1900.0 / 1920.0, 500.0 / 1080.0, InputBackendId::Normal);
    // Small delta (+50) lands at x=1950 in the void -> clamped to DP-1's right edge!
    gcomp.inject_motion(50.0, 0.0, 50.0, 0.0, InputBackendId::Normal);
    assert_eq!(gcomp.active_output().name(), "DP-1");
    assert_eq!(
        gcomp.pointer_global_location(),
        Point::from((1920.0f64.next_down(), 500.0))
    );
}

// ============================================================================
// 8. Fullscreen Windows: Animation Blocking (Finding H), Letterboxing, Layers,
//    and PointerMotionAbsolute Constraint Bypass (Finding I)
// ============================================================================

#[test]
fn test_fullscreen_animation_blocking_letterboxing_layers_and_constraints() {
    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);
    let mut client = comp.new_wayland_client();

    // Map a window with a 1280x720 buffer (smaller than 1920x1080 output so letterboxing applies in fullscreen)
    let win = comp.map_xdg_window(
        &mut client,
        Point::from((100, 100)),
        Size::from((1280, 720)),
        false,
        None,
    );

    // Map a Layer::Top panel (0, 0, 1920, 250) and a Layer::Overlay OSD (0, 0, 200, 200)
    let top_layer = comp.map_layer_surface(
        &mut client,
        Layer::Top,
        (1920, 250),
        Anchor::Top | Anchor::Left | Anchor::Right,
    );
    let overlay_layer = comp.map_layer_surface(
        &mut client,
        Layer::Overlay,
        (200, 200),
        Anchor::Top | Anchor::Left,
    );

    // Put `win` into fullscreen on DP-1
    {
        let mut shell = comp.state.common.shell.write();
        let output = shell.seats.last_active().active_output();
        let target = shell.fullscreen_request(
            &win.mapped.active_window(),
            output,
            &comp.state.common.event_loop_handle,
        );
        let seat = shell.seats.last_active().clone();
        std::mem::drop(shell);
        if let Some(target) = target {
            crate::shell::Shell::set_focus(&mut comp.state, Some(&target), &seat, None, false);
        }
    }
    comp.roundtrip(&mut client);

    // 1. Finding H: Immediately after map_fullscreen, `FullscreenSurface::is_animating()` is true (for 200ms).
    // While `is_animating()` is true, `Workspace::toplevel_surface_under` (src/shell/workspace.rs:943) returns None!
    let output = comp.active_output();
    let hit_during_anim = comp.surface_under(Point::from((960.0, 540.0)), &output);
    assert!(
        hit_during_anim.is_none(),
        "Finding H verified: Fullscreen surface is invisible to State::surface_under while FullscreenSurface::is_animating() is true"
    );

    // Wait for FULLSCREEN_ANIMATION_DURATION (200ms in src/shell/workspace.rs:75) to elapse and call update_animations()
    std::thread::sleep(Duration::from_millis(220));
    {
        let mut shell = comp.state.common.shell.write();
        shell.update_animations();
    }

    // 2. TC-FS-02: Letterboxed 1280x720 fullscreen buffer on 1920x1080 output is centered at ((1920-1280)/2, (1080-720)/2) = (320, 180)
    let hit_after_anim = comp
        .surface_under(Point::from((500.0, 500.0)), &output)
        .expect("Fullscreen surface should be hit after animation finishes");
    assert_eq!(
        hit_after_anim.0,
        PointerFocusTarget::WlSurface {
            surface: win.server_surface.clone(),
            toplevel: Some(win.mapped.active_window().into()),
        }
    );
    assert_eq!(
        hit_after_anim.1,
        Point::from((320.0, 180.0)),
        "Fullscreen 1280x720 buffer on 1920x1080 output must be centered at (320, 180)"
    );

    // 3. TC-FS-03: Layer::Top (covering y in [0..250]) is suppressed by the fullscreen window:
    // - At (960, 20) inside the top black letterbox bar (y < 180), Layer::Top is still bypassed so surface_under is None.
    // - At (960, 200) where Layer::Top and the centered fullscreen buffer [320..1600, 180..900] overlap, the fullscreen surface wins!
    assert!(
        comp.surface_under(Point::from((960.0, 20.0)), &output).is_none(),
        "Layer::Top must be suppressed in fullscreen mode even over the black letterbox area"
    );
    let hit_top_bar = comp
        .surface_under(Point::from((960.0, 200.0)), &output)
        .expect("Should hit fullscreen surface where it overlaps Layer::Top");
    assert_eq!(
        hit_top_bar.0.wl_surface().unwrap().id().protocol_id(),
        win.client_surface.id().protocol_id(),
        "Fullscreen window must take pointer priority over Layer::Top"
    );
    assert_ne!(
        hit_top_bar.0.wl_surface().unwrap().id().protocol_id(),
        top_layer.client_surface.id().protocol_id()
    );

    let hit_overlay = comp
        .surface_under(Point::from((50.0, 50.0)), &output)
        .expect("Should hit Layer::Overlay surface above fullscreen");
    assert_eq!(
        hit_overlay.0.wl_surface().unwrap().id().protocol_id(),
        overlay_layer.client_surface.id().protocol_id(),
        "Layer::Overlay must take pointer priority over fullscreen window"
    );

    // 4. TC-FS-05 & Finding I: Lock pointer on fullscreen surface, verify relative motion stays locked,
    // and verify that PointerMotionAbsolute bypasses the lock constraint!
    comp.inject_motion_absolute(500.0 / 1920.0, 500.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();
    let locked_ptr = constraints.lock_pointer(
        &win.client_surface,
        &pointer,
        None,
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    assert_eq!(client.state.locked_count, 1);

    // Relative motion keeps pointer locked at (500, 500)
    comp.inject_motion(80.0, -40.0, 80.0, -40.0, InputBackendId::Normal);
    let locked_pos = comp.pointer_global_location();
    assert!((locked_pos.x - 500.0).abs() < 1e-3 && (locked_pos.y - 500.0).abs() < 1e-3);

    // Finding I: PointerMotionAbsolute does NOT check `with_pointer_constraint` in src/input/mod.rs:707-797,
    // so injecting absolute motion moves the cursor away even while Locked!
    comp.inject_motion_absolute(900.0 / 1920.0, 800.0 / 1080.0, InputBackendId::Normal);
    let bypassed_pos = comp.pointer_global_location();
    assert!(
        (bypassed_pos.x - 900.0).abs() < 1e-3 && (bypassed_pos.y - 800.0).abs() < 1e-3,
        "Finding I verified: PointerMotionAbsolute bypasses active PointerConstraint::Locked"
    );

    locked_ptr.destroy();
}

// ============================================================================
// 9. Cross-Monitor Spanning Windows: Hit-Testing, Confinement, Warp (Finding G),
//    and Interactive MoveGrab Migration
// ============================================================================

#[test]
fn test_cross_monitor_spanning_windows_hit_testing_confine_warp_and_move_grab() {
    // Dual 1920x1080 monitors side-by-side: DP-1 [0..1920, 0..1080], DP-2 [1920..3840, 0..1080]
    let mut comp = TestCompositor::new(&[
        ("DP-1", (1920, 1080), 1.0, (0, 0)),
        ("DP-2", (1920, 1080), 1.0, (1920, 0)),
    ]);
    let mut client = comp.new_wayland_client();

    // Map an 800x600 floating window on DP-1 at (1520, 200), so it spans [1520..2320, 200..800]:
    // - [1520..1920] (width 400) is on DP-1
    // - [1920..2320] (width 400) overhangs onto DP-2
    let win = comp.map_xdg_window(
        &mut client,
        Point::from((1520, 200)),
        Size::from((800, 600)),
        false,
        None,
    );

    let outputs = comp.outputs();
    let dp1 = outputs[0].clone();
    let dp2 = outputs[1].clone();

    // 1. TC-SPAN-02: Hit-testing on DP-1 portion (1700, 400) succeeds;
    // hit-testing on DP-2 overhanging portion (2050, 400) returns None!
    let hit_dp1 = comp.surface_under(Point::from((1700.0, 400.0)), &dp1);
    assert!(hit_dp1.is_some(), "Portion of window on DP-1 must be hit");
    assert_eq!(hit_dp1.unwrap().1, Point::from((1520.0, 200.0)));

    let hit_dp2_via_dp2 = comp.surface_under(Point::from((2050.0, 400.0)), &dp2);
    let hit_dp2_via_dp1 = comp.surface_under(Point::from((2050.0, 400.0)), &dp1);
    assert!(
        hit_dp2_via_dp2.is_none() && hit_dp2_via_dp1.is_none(),
        "TC-SPAN-02 verified: Overhanging portion of floating window on DP-2 is unreachable by State::surface_under"
    );

    // 2. TC-SPAN-04 & Finding G: Warping pointer into the overhanging portion on DP-2 via wp_pointer_warp_v1 / apply_cursor_hint
    comp.inject_motion_absolute(1700.0 / 1920.0, 400.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    let enter_serial = client.state.last_enter_serial().unwrap();

    client.state.clear_events();
    // Warp to window-local (600, 200) -> global (1520 + 600, 200 + 200) = (2120, 400) on DP-2!
    // Because 600 <= 800, `in_window` in `apply_cursor_hint` is true!
    let warp_mgr = client.state.pointer_warp.as_ref().unwrap().clone();
    let pointer = client.state.pointer.as_ref().unwrap().clone();
    warp_mgr.warp_pointer(&win.client_surface, &pointer, 600.0, 200.0, enter_serial);
    comp.roundtrip(&mut client);

    // Verify Finding G:
    // (a) Global pointer position moved to (2120, 400) on DP-2
    assert_eq!(comp.pointer_global_location(), Point::from((2120.0, 400.0)));
    // (b) Because `State::surface_under((2120, 400), &DP-1)` returned None, the window received `wl_pointer.leave`!
    assert_eq!(
        client.state.pointer_leaves.len(),
        1,
        "Finding G verified: Warping into overhanging cross-monitor region causes immediate wl_pointer.leave"
    );
    // (c) `seat.active_output()` was NOT updated to DP-2 and remains DP-1!
    assert_eq!(
        comp.active_output().name(),
        "DP-1",
        "Finding G verified: apply_cursor_hint does not update seat.active_output() when warping across monitors"
    );

    // 3. TC-SPAN-03: Confine pointer on the cross-monitor window at (1800, 400) on DP-1,
    // then move +200px right into the overhanging part (2000, 400) on DP-2.
    comp.inject_motion_absolute(1800.0 / 1920.0, 400.0 / 1080.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
    let confined_ptr = constraints.confine_pointer(
        &win.client_surface,
        &pointer,
        None,
        zwp_pointer_constraints_v1::Lifetime::Oneshot,
        &client.qh,
        (),
    );
    comp.roundtrip(&mut client);
    assert_eq!(client.state.confined_count, 1);

    // Move +200px right toward (2000, 400) on DP-2:
    // Because `is_legal` (src/input/mod.rs:562) calls `State::surface_under((2000, 400), &DP-2, shell)`,
    // which returns `None` (since the window belongs to DP-1's workspace), `is_legal` rejects the move,
    // reverts `position` to `original_position` (1800, 400), and returns early at line 638!
    // Thus, PointerConstraint::Confined blocks the cursor at the monitor boundary and prevents it from
    // entering the visible right half of the cross-monitor window on DP-2!
    comp.inject_motion(200.0, 0.0, 200.0, 0.0, InputBackendId::Normal);
    comp.flush_to_client(&mut client);
    assert_eq!(
        comp.active_output().name(),
        "DP-1",
        "PointerConstraint::Confined blocks cursor from crossing onto DP-2 part of a spanning window"
    );
    assert!((comp.pointer_global_location().x - 1800.0).abs() < 1e-6);
    assert!((comp.pointer_global_location().y - 400.0).abs() < 1e-6);
    confined_ptr.destroy();
    comp.roundtrip(&mut client);

    // 4. TC-SPAN-01: Interactive MoveGrab dragging window from DP-1 across to DP-2
    {
        let shell = comp.state.common.shell.read();
        let seat = shell.seats.last_active().clone();
        seat.set_active_output(&dp1);
    }
    comp.inject_motion_absolute(1700.0 / 1920.0, 400.0 / 1080.0, InputBackendId::Normal);
    comp.inject_button(0x110, ButtonState::Pressed, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    // Initiate server-side MoveGrab (as triggered by xdg_toplevel.move or Super+LeftClick)
    {
        let mut shell = comp.state.common.shell.write();
        let seat = shell.seats.last_active().clone();
        let res = shell.move_request(
            &win.server_surface,
            &seat,
            None,
            crate::shell::grabs::ReleaseMode::NoMouseButtons,
            false,
            &comp.state.common.config,
            &comp.state.common.event_loop_handle,
            false,
        );
        std::mem::drop(shell);
        if let Some((grab, focus)) = res {
            let ptr = seat.get_pointer().unwrap();
            ptr.set_grab(
                &mut comp.state,
                grab,
                SERIAL_COUNTER.next_serial(),
                focus,
            );
        }
    }

    // Drag pointer by +600px in X across monitor boundary to (2300, 400) on DP-2 and release button
    comp.inject_motion(600.0, 0.0, 600.0, 0.0, InputBackendId::Normal);
    comp.inject_button(0x110, ButtonState::Released, InputBackendId::Normal);
    comp.flush_to_client(&mut client);

    assert_eq!(comp.active_output().name(), "DP-2");
    // Now the window has been migrated to DP-2's active workspace, so hit-testing on DP-2 at (2300, 400) hits it!
    let hit_after_move = comp.surface_under(Point::from((2300.0, 400.0)), &dp2);
    assert!(
        hit_after_move.is_some(),
        "After MoveGrab across monitors, window is migrated to DP-2 and receives pointer hits on DP-2"
    );
}

// ============================================================================
// 10. X11 / XWayland Integration: XwaylandDescaling Modes, Managed Windows,
//     and Cross-Monitor X11 Override-Redirect Windows (TC-SPAN-05)
// ============================================================================

#[test]
fn test_xwayland_scaling_modes_and_cross_monitor_override_redirect_windows() {
    // Dual monitors: DP-1 (1920x1080 @ 2.0 -> logical 960x540 at (0, 0)),
    //                DP-2 (1920x1080 @ 1.0 -> logical 1920x1080 at (960, 0))
    let mut comp = TestCompositor::new(&[
        ("DP-1", (1920, 1080), 2.0, (0, 0)),
        ("DP-2", (1920, 1080), 1.0, (960, 0)),
    ]);

    // Launch real rootless Xwayland via cosmic-comp's State::launch_xwayland
    comp.state.launch_xwayland(None);

    for _ in 0..200 {
        comp.dispatch();
        if comp
            .state
            .common
            .xwayland_state
            .as_ref()
            .and_then(|s| s.xwm.as_ref())
            .is_some()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let xw_display = comp
        .state
        .common
        .xwayland_state
        .as_ref()
        .expect("Xwayland state should be initialized")
        .display;

    // 1. Verify XwaylandDescaling modes (Disabled, Enabled, Fractional) on scale=2.0 primary output
    comp.state.common.config.cosmic_conf.descale_xwayland = XwaylandDescaling::Disabled;
    comp.state.common.update_xwayland_settings();
    assert_eq!(comp.state.common.xwayland_scale, Some(1.0));
    let xw_client = comp.state.common.xwayland_state.as_ref().unwrap().client.clone();
    assert_eq!(client_compositor_state(&xw_client).client_scale(), 1.0);

    comp.state.common.config.cosmic_conf.descale_xwayland = XwaylandDescaling::Enabled;
    comp.state.common.update_xwayland_settings();
    assert_eq!(comp.state.common.xwayland_scale, Some(2.0));
    assert_eq!(client_compositor_state(&xw_client).client_scale(), 2.0);

    // Reset to Disabled (scale 1.0) for 1:1 coordinate verification on cross-monitor windows
    comp.state.common.config.cosmic_conf.descale_xwayland = XwaylandDescaling::Disabled;
    comp.state.common.update_xwayland_settings();

    // 2. Connect real X11 client to cosmic-comp's Xwayland instance
    let (x11_conn, screen_num) =
        smithay::reexports::x11rb::connect(Some(&format!(":{}", xw_display)))
            .expect("Failed to connect X11 client to Xwayland");
    let screen = &x11_conn.setup().roots[screen_num];
    let root = screen.root;

    // Create an Override-Redirect X11 window at (700, 100) with size (600, 300),
    // which spans across DP-1 [0..960] and DP-2 [960..2880]: [700..1300, 100..400]!
    let or_win = x11_conn.generate_id().unwrap();
    let gc = x11_conn.generate_id().unwrap();
    x11_conn
        .create_gc(gc, root, &Default::default())
        .unwrap();
    x11_conn
        .create_window(
            screen.root_depth,
            or_win,
            root,
            700,
            100,
            600,
            300,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new()
                .background_pixel(screen.white_pixel)
                .override_redirect(1)
                .event_mask(
                    EventMask::POINTER_MOTION
                        | EventMask::ENTER_WINDOW
                        | EventMask::LEAVE_WINDOW
                        | EventMask::EXPOSURE
                        | EventMask::STRUCTURE_NOTIFY,
                ),
        )
        .unwrap();
    x11_conn.map_window(or_win).unwrap();
    x11_conn.flush().unwrap();

    // Pump event loop until X11Wm pairs the Override-Redirect X11Surface with its WlSurface
    for _ in 0..200 {
        let _ = x11_conn.clear_area(true, or_win, 0, 0, 600, 300);
        let _ = x11_conn.flush();
        comp.dispatch();
        if comp
            .state
            .common
            .shell
            .read()
            .override_redirect_windows
            .iter()
            .any(|or| or.window_id() == or_win && or.wl_surface().is_some())
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let has_or = comp
        .state
        .common
        .shell
        .read()
        .override_redirect_windows
        .iter()
        .any(|or| or.window_id() == or_win && or.wl_surface().is_some());
    assert!(
        has_or,
        "X11 override-redirect window must be paired with WlSurface in Shell::override_redirect_windows"
    );

    // 3. TC-SPAN-05: Verify that the cross-monitor X11 override-redirect window [700..1300, 100..400]
    // is hit by `State::surface_under` on BOTH DP-1 (at 800, 200) AND DP-2 (at 1100, 200)!
    let outputs = comp.outputs();
    let dp1 = outputs[0].clone();
    let dp2 = outputs[1].clone();

    let hit_or_dp1 = comp
        .surface_under(Point::from((800.0, 200.0)), &dp1)
        .expect("Override-redirect X11 window should be hit on DP-1");
    assert_eq!(hit_or_dp1.1, Point::from((700.0, 100.0)));

    let hit_or_dp2 = comp
        .surface_under(Point::from((1100.0, 200.0)), &dp2)
        .expect("Override-redirect X11 window should ALSO be hit on DP-2");
    assert_eq!(hit_or_dp2.1, Point::from((700.0, 100.0)));

    // Inject pointer motion onto the X11 window on DP-1 (800, 200) and then across onto DP-2 (1100, 200)
    // and verify the X11 client receives real X11 EnterNotify / MotionNotify events from Xwayland!
    comp.inject_motion_absolute(800.0 / 960.0, 200.0 / 540.0, InputBackendId::Normal);
    comp.inject_motion(10.0, 5.0, 10.0, 5.0, InputBackendId::Normal);
    for _ in 0..20 {
        comp.dispatch();
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut received_x11_pointer_event = false;
    while let Ok(Some(evt)) = x11_conn.poll_for_event() {
        match evt {
            smithay::reexports::x11rb::protocol::Event::EnterNotify(e) if e.event == or_win => {
                received_x11_pointer_event = true;
            }
            smithay::reexports::x11rb::protocol::Event::MotionNotify(m) if m.event == or_win => {
                received_x11_pointer_event = true;
            }
            _ => {}
        }
    }
    assert!(
        received_x11_pointer_event,
        "X11 client must receive real EnterNotify/MotionNotify events via Xwayland"
    );

    // 4. Also create and map a managed (non-override-redirect) X11 toplevel window
    let managed_win = x11_conn.generate_id().unwrap();
    x11_conn
        .create_window(
            screen.root_depth,
            managed_win,
            root,
            100,
            100,
            300,
            200,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new()
                .background_pixel(screen.white_pixel)
                .override_redirect(0)
                .event_mask(
                    EventMask::POINTER_MOTION
                        | EventMask::ENTER_WINDOW
                        | EventMask::LEAVE_WINDOW
                        | EventMask::EXPOSURE
                        | EventMask::STRUCTURE_NOTIFY,
                ),
        )
        .unwrap();
    x11_conn.unmap_window(or_win).unwrap();
    x11_conn.map_window(managed_win).unwrap();
    x11_conn.flush().unwrap();

    for _ in 0..200 {
        let _ = x11_conn.clear_area(true, managed_win, 0, 0, 300, 200);
        let _ = x11_conn.flush();
        comp.dispatch();
        if comp
            .state
            .common
            .shell
            .read()
            .mapped()
            .any(|m| m.active_window().x11_surface().is_some_and(|x| x.window_id() == managed_win))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mapped_x11 = comp
        .state
        .common
        .shell
        .read()
        .mapped()
        .find(|m| {
            m.active_window()
                .x11_surface()
                .is_some_and(|x| x.window_id() == managed_win)
        })
        .cloned();
    assert!(
        mapped_x11.is_some(),
        "Managed X11 window must be mapped into Shell via X11WmHandler::mapped_window_notify"
    );
}

// ============================================================================
// 11. Wayland Surface: Windowed Mode -> Fullscreen Mode -> Windowed Mode
//     Across Resolutions & Scales (Pointer Motion, Relative Motion, Lock & Unlock)
// ============================================================================

#[test]
fn test_wayland_surface_windowed_to_fullscreen_across_resolutions_lock_unlock_motion_and_relative() {
    let configs = [
        ("FHD-100", (1920, 1080), 1.0),
        ("QHD-125", (2560, 1440), 1.25),
        ("UHD-150", (3840, 2160), 1.5),
        ("UHD-200", (3840, 2160), 2.0),
    ];

    for (name, (phys_w, phys_h), scale) in configs {
        let mut comp = TestCompositor::new(&[(name, (phys_w, phys_h), scale, (0, 0))]);
        let mut client = comp.new_wayland_client();

        let output = comp.active_output();
        let out_geo = output.geometry();
        let out_w = out_geo.size.w;
        let out_h = out_geo.size.h;

        // --------------------------------------------------------------------
        // Phase 1: Windowed Mode (800x600 at (100, 100))
        // --------------------------------------------------------------------
        let win = comp.map_xdg_window(
            &mut client,
            Point::from((100, 100)),
            Size::from((800, 600)),
            false,
            None,
        );
        client.state.clear_events();

        // 1A. Unlocked PointerMotionAbsolute & PointerMotion in Windowed Mode
        comp.inject_motion_absolute(
            250.0 / (out_w as f64),
            220.0 / (out_h as f64),
            InputBackendId::Normal,
        );
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 250.0).abs() < 1e-3 && (pos.y - 220.0).abs() < 1e-3);
        assert_eq!(client.state.pointer_enters.len(), 1);
        assert!(
            (client.state.pointer_enters[0].1 - 150.0).abs() < 1e-2
                && (client.state.pointer_enters[0].2 - 120.0).abs() < 1e-2,
            "[{name}] Windowed enter surface-local coords must be (150, 120)"
        );
        client.state.clear_events();

        comp.inject_motion(24.5, -12.25, 20.0, -10.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 274.5).abs() < 1e-3 && (pos.y - 207.75).abs() < 1e-3);
        assert_eq!(client.state.pointer_motions.len(), 1);
        assert!(
            (client.state.pointer_motions[0].0 - 174.5).abs() < 1e-2
                && (client.state.pointer_motions[0].1 - 107.75).abs() < 1e-2,
            "[{name}] Windowed motion surface-local coords must be (174.5, 107.75)"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![(24.5, -12.25, 20.0, -10.0)],
            "[{name}] Windowed relative_motion must match injected (dx, dy, dx_unaccel, dy_unaccel)"
        );
        client.state.clear_events();

        // 1B. Pointer Lock in Windowed Mode
        let constraints = client.state.pointer_constraints.as_ref().unwrap().clone();
        let pointer = client.state.pointer.as_ref().unwrap().clone();
        let locked_win_ptr = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Oneshot,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        assert_eq!(
            client.state.locked_count, 1,
            "[{name}] Windowed lock_pointer must emit locked event"
        );
        client.state.clear_events();

        comp.inject_motion(60.0, 45.0, 50.0, 35.0, InputBackendId::Normal);
        comp.inject_motion(-150.0, 90.0, -120.0, 75.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let locked_pos = comp.pointer_global_location();
        assert!(
            (locked_pos.x - 274.5).abs() < 1e-3 && (locked_pos.y - 207.75).abs() < 1e-3,
            "[{name}] Windowed locked pointer must freeze global cursor location"
        );
        assert!(
            client.state.pointer_motions.is_empty(),
            "[{name}] Windowed locked pointer must suppress wl_pointer.motion"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![
                (60.0, 45.0, 50.0, 35.0),
                (-150.0, 90.0, -120.0, 75.0),
            ],
            "[{name}] Windowed locked pointer must still deliver all relative_motion events"
        );
        client.state.clear_events();

        // 1C. Pointer Unlock + Cursor Position Hint in Windowed Mode
        locked_win_ptr.set_cursor_position_hint(320.0, 240.0);
        win.client_surface.commit();
        comp.roundtrip(&mut client);

        locked_win_ptr.destroy();
        comp.roundtrip(&mut client);

        let unlocked_pos = comp.pointer_global_location();
        assert!(
            (unlocked_pos.x - 420.0).abs() < 1e-3 && (unlocked_pos.y - 340.0).abs() < 1e-3,
            "[{name}] Windowed unlock with hint (320, 240) at window (100, 100) must warp global cursor to (420, 340)"
        );
        client.state.clear_events();

        // Post-unlock motion in Windowed Mode
        comp.inject_motion(18.25, -14.5, 15.0, -12.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 438.25).abs() < 1e-3 && (pos.y - 325.5).abs() < 1e-3);
        assert_eq!(client.state.pointer_motions.len(), 1);
        assert!(
            (client.state.pointer_motions[0].0 - 338.25).abs() < 1e-2
                && (client.state.pointer_motions[0].1 - 225.5).abs() < 1e-2,
            "[{name}] Windowed post-unlock wl_pointer.motion must resume"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![(18.25, -14.5, 15.0, -12.0)],
            "[{name}] Windowed post-unlock relative_motion must resume"
        );
        client.state.clear_events();

        // --------------------------------------------------------------------
        // Phase 2: Transition to Fullscreen Mode via xdg_toplevel.set_fullscreen
        //          (XdgShellHandler::fullscreen_request -> Shell::fullscreen_request)
        // --------------------------------------------------------------------
        win.xdg_toplevel.set_fullscreen(None);
        comp.roundtrip(&mut client);

        // Wait for FULLSCREEN_ANIMATION_DURATION (200ms) and update animations
        std::thread::sleep(Duration::from_millis(220));
        {
            let mut shell = comp.state.common.shell.write();
            shell.update_animations();
        }

        // Sub-phase 2A: Fullscreen with letterboxed 800x600 buffer centered at ((out_w - 800)/2, (out_h - 600)/2)
        let letterbox_x = ((out_w - 800) / 2) as f64;
        let letterbox_y = ((out_h - 600) / 2) as f64;
        let hit_fs_letterbox = comp
            .surface_under(Point::from((letterbox_x + 200.0, letterbox_y + 150.0)), &output)
            .expect("Letterboxed fullscreen Wayland surface must be hit after animation");
        assert_eq!(
            hit_fs_letterbox.1,
            Point::from((letterbox_x, letterbox_y)),
            "[{name}] Letterboxed 800x600 fullscreen surface origin must be centered"
        );

        // Sub-phase 2B: Client commits full-output-sized buffer (out_w, out_h) for native Fullscreen
        let fs_buf = comp.create_shm_buffer(&mut client, out_w, out_h);
        win.xdg_surface.set_window_geometry(0, 0, out_w, out_h);
        win.client_surface.attach(Some(&fs_buf), 0, 0);
        win.client_surface.damage_buffer(0, 0, out_w, out_h);
        win.client_surface.commit();
        comp.roundtrip(&mut client);
        comp.state.common.refresh();

        let hit_fs_full = comp
            .surface_under(Point::from((500.0, 400.0)), &output)
            .expect("Full-output fullscreen Wayland surface must be hit");
        assert_eq!(
            hit_fs_full.1,
            Point::from((0.0, 0.0)),
            "[{name}] Full-output fullscreen surface origin must be (0, 0)"
        );
        client.state.clear_events();

        // 2B-1: Unlocked PointerMotionAbsolute & PointerMotion in Fullscreen Mode
        comp.inject_motion_absolute(
            500.0 / (out_w as f64),
            400.0 / (out_h as f64),
            InputBackendId::Normal,
        );
        comp.flush_to_client(&mut client);
        client.state.clear_events();

        comp.inject_motion(32.5, -22.75, 28.0, -19.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 532.5).abs() < 1e-3 && (pos.y - 377.25).abs() < 1e-3);
        assert_eq!(client.state.pointer_motions.len(), 1);
        assert!(
            (client.state.pointer_motions[0].0 - 532.5).abs() < 1e-2
                && (client.state.pointer_motions[0].1 - 377.25).abs() < 1e-2,
            "[{name}] Fullscreen unlocked wl_pointer.motion must equal global logical coordinates (532.5, 377.25)"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![(32.5, -22.75, 28.0, -19.0)],
            "[{name}] Fullscreen unlocked relative_motion must match injected deltas"
        );
        client.state.clear_events();

        // 2B-2: Pointer Lock in Fullscreen Mode
        let locked_fs_ptr = constraints.lock_pointer(
            &win.client_surface,
            &pointer,
            None,
            zwp_pointer_constraints_v1::Lifetime::Oneshot,
            &client.qh,
            (),
        );
        comp.roundtrip(&mut client);
        assert_eq!(
            client.state.locked_count, 2,
            "[{name}] Fullscreen lock_pointer must emit locked event"
        );
        client.state.clear_events();

        comp.inject_motion(350.0, -250.0, 300.0, -210.0, InputBackendId::Normal);
        comp.inject_motion(-180.0, 120.0, -150.0, 100.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let fs_locked_pos = comp.pointer_global_location();
        assert!(
            (fs_locked_pos.x - 532.5).abs() < 1e-3 && (fs_locked_pos.y - 377.25).abs() < 1e-3,
            "[{name}] Fullscreen locked pointer must stay frozen at (532.5, 377.25)"
        );
        assert!(
            client.state.pointer_motions.is_empty(),
            "[{name}] Fullscreen locked pointer must suppress wl_pointer.motion"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![
                (350.0, -250.0, 300.0, -210.0),
                (-180.0, 120.0, -150.0, 100.0),
            ],
            "[{name}] Fullscreen locked pointer must deliver all relative_motion events"
        );
        client.state.clear_events();

        // 2B-3: Pointer Unlock + Cursor Position Hint in Fullscreen Mode
        locked_fs_ptr.set_cursor_position_hint(700.0, 500.0);
        win.client_surface.commit();
        comp.roundtrip(&mut client);

        locked_fs_ptr.destroy();
        comp.roundtrip(&mut client);

        let fs_unlocked_pos = comp.pointer_global_location();
        assert!(
            (fs_unlocked_pos.x - 700.0).abs() < 1e-3 && (fs_unlocked_pos.y - 500.0).abs() < 1e-3,
            "[{name}] Fullscreen unlock with hint (700, 500) must warp global cursor to (700, 500)"
        );
        client.state.clear_events();

        // Post-unlock motion in Fullscreen Mode
        comp.inject_motion(-40.0, 30.0, -32.0, 24.0, InputBackendId::Normal);
        comp.flush_to_client(&mut client);

        let pos = comp.pointer_global_location();
        assert!((pos.x - 660.0).abs() < 1e-3 && (pos.y - 530.0).abs() < 1e-3);
        assert_eq!(client.state.pointer_motions.len(), 1);
        assert!(
            (client.state.pointer_motions[0].0 - 660.0).abs() < 1e-2
                && (client.state.pointer_motions[0].1 - 530.0).abs() < 1e-2,
            "[{name}] Fullscreen post-unlock wl_pointer.motion must resume"
        );
        assert_eq!(
            client.state.relative_motions,
            vec![(-40.0, 30.0, -32.0, 24.0)],
            "[{name}] Fullscreen post-unlock relative_motion must resume"
        );

        // --------------------------------------------------------------------
        // Phase 3: Unfullscreen (xdg_toplevel.unset_fullscreen) back to Windowed Mode
        // --------------------------------------------------------------------
        win.xdg_toplevel.unset_fullscreen();
        comp.roundtrip(&mut client);

        let win_buf = comp.create_shm_buffer(&mut client, 800, 600);
        win.xdg_surface.set_window_geometry(0, 0, 800, 600);
        win.client_surface.attach(Some(&win_buf), 0, 0);
        win.client_surface.damage_buffer(0, 0, 800, 600);
        win.client_surface.commit();
        comp.roundtrip(&mut client);
        comp.state.common.refresh();

        let hit_restored = comp
            .surface_under(Point::from((250.0, 220.0)), &output)
            .expect("Unfullscreened Wayland window must be hit in Windowed mode");
        assert_eq!(
            hit_restored.1,
            Point::from((100.0, 100.0)),
            "[{name}] Unfullscreened Wayland window must restore its Windowed origin (100, 100)"
        );
    }
}

// ============================================================================
// 12. X11 Surface: Windowed Mode -> Fullscreen Mode -> Windowed Mode
//     Across Resolutions & Scales (Pointer Motion, Relative Motion, Lock & Unlock)
// ============================================================================

#[test]
fn test_x11_surface_windowed_to_fullscreen_across_resolutions_lock_unlock_motion_and_relative() {
    use crate::input::PointerConstraint;
    use smithay::reexports::x11rb::{
        connection::Connection as _,
        protocol::{
            xfixes::ConnectionExt as _,
            xproto::{
                ClientMessageData, ClientMessageEvent, ConnectionExt as _, CreateWindowAux,
                EventMask, GrabMode, WindowClass, CLIENT_MESSAGE_EVENT,
            },
        },
    };

    let mut comp = TestCompositor::new(&[("DP-1", (1920, 1080), 1.0, (0, 0))]);

    // Launch real rootless Xwayland via cosmic-comp's State::launch_xwayland
    comp.state.launch_xwayland(None);
    for _ in 0..200 {
        comp.dispatch();
        if comp
            .state
            .common
            .xwayland_state
            .as_ref()
            .and_then(|s| s.xwm.as_ref())
            .is_some()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let xw_display = comp
        .state
        .common
        .xwayland_state
        .as_ref()
        .expect("Xwayland state should be initialized")
        .display;

    let (x11_conn, screen_num) =
        smithay::reexports::x11rb::connect(Some(&format!(":{}", xw_display)))
            .expect("Failed to connect X11 client to Xwayland");
    x11_conn.xfixes_query_version(5, 0).unwrap().reply().unwrap();
    let screen = &x11_conn.setup().roots[screen_num];
    let root = screen.root;

    let net_wm_state_atom = x11_conn
        .intern_atom(false, b"_NET_WM_STATE")
        .unwrap()
        .reply()
        .unwrap()
        .atom;
    let net_wm_state_fullscreen_atom = x11_conn
        .intern_atom(false, b"_NET_WM_STATE_FULLSCREEN")
        .unwrap()
        .reply()
        .unwrap()
        .atom;

    // Create and map a managed X11 window (600x400)
    let x11_win = x11_conn.generate_id().unwrap();
    x11_conn
        .create_window(
            screen.root_depth,
            x11_win,
            root,
            100,
            100,
            600,
            400,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new()
                .background_pixel(screen.white_pixel)
                .override_redirect(0)
                .event_mask(
                    EventMask::POINTER_MOTION
                        | EventMask::ENTER_WINDOW
                        | EventMask::LEAVE_WINDOW
                        | EventMask::EXPOSURE
                        | EventMask::STRUCTURE_NOTIFY,
                ),
        )
        .unwrap();
    x11_conn.map_window(x11_win).unwrap();
    x11_conn.flush().unwrap();

    for _ in 0..200 {
        let _ = x11_conn.clear_area(true, x11_win, 0, 0, 600, 400);
        let _ = x11_conn.flush();
        comp.dispatch();
        if comp
            .state
            .common
            .shell
            .read()
            .mapped()
            .any(|m| m.active_window().x11_surface().is_some_and(|x| x.window_id() == x11_win))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let mapped_x11 = comp
        .state
        .common
        .shell
        .read()
        .mapped()
        .find(|m| {
            m.active_window()
                .x11_surface()
                .is_some_and(|x| x.window_id() == x11_win)
        })
        .cloned()
        .expect("Managed X11 window must be mapped in Shell");
    let x11_surface = mapped_x11
        .active_window()
        .x11_surface()
        .unwrap()
        .clone();
    let x11_wl_surface = x11_surface
        .wl_surface()
        .expect("X11Surface must be paired with WlSurface");

    // Test matrix of (name, physical resolution, output scale, XwaylandDescaling mode, expected client_scale)
    let configs = [
        ("FHD-100-Disabled", (1920, 1080), 1.0, XwaylandDescaling::Disabled, 1.0),
        ("QHD-125-Fractional", (2560, 1440), 1.25, XwaylandDescaling::Fractional, 1.25),
        ("UHD-150-Fractional", (3840, 2160), 1.5, XwaylandDescaling::Fractional, 1.5),
        ("UHD-200-Enabled", (3840, 2160), 2.0, XwaylandDescaling::Enabled, 2.0),
    ];

    for (name, (phys_w, phys_h), scale, descaling, expected_client_scale) in configs {
        let output = comp.active_output();
        let mode = OutputMode {
            size: (phys_w, phys_h).into(),
            refresh: 60_000,
        };
        output.add_mode(mode);
        output.set_preferred(mode);
        output.change_current_state(
            Some(mode),
            Some(Transform::Normal),
            Some(Scale::Fractional(scale)),
            Some((0, 0).into()),
        );
        {
            let mut cfg = output.config_mut();
            cfg.mode = ((phys_w, phys_h), Some(60_000));
            cfg.scale = scale;
            cfg.position = (0, 0);
        }
        comp.state.common.config.cosmic_conf.descale_xwayland = descaling;
        comp.state.common.refresh();
        comp.state.common.update_xwayland_primary_output();
        comp.state.common.update_xwayland_settings();

        let xw_client = comp.state.common.xwayland_state.as_ref().unwrap().client.clone();
        assert_eq!(
            client_compositor_state(&xw_client).client_scale(),
            expected_client_scale,
            "[{name}] Xwayland client_scale mismatch"
        );

        let out_geo = output.geometry();
        let out_w = out_geo.size.w;
        let out_h = out_geo.size.h;

        // --------------------------------------------------------------------
        // Phase 1: X11 Surface in Windowed Mode (at (100, 100), 600x400 + 32px SSD)
        // --------------------------------------------------------------------
        {
            let mut shell = comp.state.common.shell.write();
            let ws = shell.workspaces.active_mut(&output).unwrap();
            mapped_x11.set_geometry(
                Rectangle::new(Point::from((100, 100)), Size::from((600, 436))).to_global(&output),
            );
            ws.floating_layer
                .space
                .map_element(mapped_x11.clone(), Point::from((100, 100)), false);
            ws.floating_layer.space.refresh();
        }
        for _ in 0..30 {
            let _ = x11_conn.clear_area(true, x11_win, 0, 0, 600, 400);
            let _ = x11_conn.flush();
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }
        while let Ok(Some(_)) = x11_conn.poll_for_event() {}

        // Verify Windowed SSD headerbar at (250, 115) vs X11Surface client area at (250, 236) (surface origin (100, 136))
        let hit_ssd = comp
            .surface_under(Point::from((250.0, 115.0)), &output)
            .expect("Should hit SSD headerbar of windowed X11 window");
        assert!(
            matches!(hit_ssd.0, PointerFocusTarget::WindowUI(_)),
            "[{name}] [100..700, 100..136] must hit CosmicWindow SSD WindowUI"
        );

        let hit_x11_win = comp
            .surface_under(Point::from((250.0, 236.0)), &output)
            .expect("Should hit X11Surface client area below 36px SSD");
        assert_eq!(
            hit_x11_win.0,
            PointerFocusTarget::X11Surface {
                surface: x11_surface.clone(),
                toplevel: Some(mapped_x11.active_window()),
            },
            "[{name}] Point below 36px SSD must hit PointerFocusTarget::X11Surface"
        );
        assert_eq!(
            hit_x11_win.1,
            Point::from((100.0, 136.0)),
            "[{name}] Windowed X11Surface origin must include +36px SSD y-offset (100, 136)"
        );

        // 1A. Unlocked PointerMotionAbsolute & PointerMotion in Windowed Mode
        comp.inject_motion_absolute(
            250.0 / (out_w as f64),
            232.0 / (out_h as f64),
            InputBackendId::Normal,
        );
        comp.inject_motion(20.0, 15.0, 16.0, 12.0, InputBackendId::Normal);
        for _ in 0..20 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let pos = comp.pointer_global_location();
        assert!((pos.x - 270.0).abs() < 1e-3 && (pos.y - 247.0).abs() < 1e-3);

        let mut win_motion_events = Vec::new();
        while let Ok(Some(evt)) = x11_conn.poll_for_event() {
            if let smithay::reexports::x11rb::protocol::Event::MotionNotify(m) = evt
                && m.event == x11_win
            {
                win_motion_events.push((m.event_x, m.event_y, m.root_x, m.root_y));
            }
        }
        assert!(
            !win_motion_events.is_empty(),
            "[{name}] Windowed X11 client must receive MotionNotify events when unlocked"
        );

        // 1B. Pointer Lock (via X11 XFixesHideCursor + XGrabPointer + XWarpPointer -> Xwayland zwp_pointer_constraints_v1.lock_pointer
        //     AND direct PointerConstraint::Locked verification on x11_wl_surface) in Windowed Mode
        x11_conn.xfixes_hide_cursor(x11_win).unwrap();
        x11_conn
            .grab_pointer(
                false,
                x11_win,
                EventMask::POINTER_MOTION,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
                x11_win,
                smithay::reexports::x11rb::NONE,
                smithay::reexports::x11rb::CURRENT_TIME,
            )
            .unwrap()
            .reply()
            .unwrap();
        x11_conn
            .warp_pointer(
                smithay::reexports::x11rb::NONE,
                x11_win,
                0,
                0,
                0,
                0,
                150,
                100,
            )
            .unwrap();
        x11_conn.flush().unwrap();
        for _ in 0..25 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let ptr = comp
            .state
            .common
            .shell
            .read()
            .seats
            .last_active()
            .get_pointer()
            .unwrap();
        let has_xw_constraint =
            smithay::wayland::pointer_constraints::with_pointer_constraint(&x11_wl_surface, &ptr, |c| {
                c.is_some_and(|c| c.is_active())
            });
        assert!(
            has_xw_constraint,
            "[{name}] Windowed X11 grab_pointer + hide_cursor + warp_pointer must activate a Wayland pointer constraint on x11_wl_surface via Xwayland"
        );

        let is_locked = smithay::wayland::pointer_constraints::with_pointer_constraint(
            &x11_wl_surface,
            &ptr,
            |c| matches!(c.as_deref(), Some(PointerConstraint::Locked(_))),
        );
        assert!(
            is_locked,
            "[{name}] Windowed X11 hide_cursor + grab_pointer + warp_pointer must create PointerConstraint::Locked on x11_wl_surface"
        );
        let lock_origin = comp.pointer_global_location();
        while let Ok(Some(_)) = x11_conn.poll_for_event() {}

        // Inject relative motion while X11 surface is Locked:
        // cosmic-comp freezes global cursor position and suppresses wl_pointer.motion,
        // while delivering zwp_relative_pointer_v1.relative_motion to Xwayland, which emits X11 MotionNotify!
        comp.inject_motion(45.0, -30.0, 40.0, -25.0, InputBackendId::Normal);
        comp.inject_motion(-80.0, 60.0, -70.0, 50.0, InputBackendId::Normal);
        for _ in 0..20 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let after_lock_motion = comp.pointer_global_location();
        assert!(
            (after_lock_motion.x - lock_origin.x).abs() < 1e-3
                && (after_lock_motion.y - lock_origin.y).abs() < 1e-3,
            "[{name}] Windowed X11 Locked pointer must freeze global cursor location in cosmic-comp"
        );

        let mut locked_win_x11_motions = Vec::new();
        while let Ok(Some(evt)) = x11_conn.poll_for_event() {
            if let smithay::reexports::x11rb::protocol::Event::MotionNotify(m) = evt
                && m.event == x11_win
            {
                locked_win_x11_motions.push((m.event_x, m.event_y));
            }
        }
        assert!(
            !locked_win_x11_motions.is_empty(),
            "[{name}] Windowed X11 client must still receive relative-motion-driven events via zwp_relative_pointer_v1 while Locked"
        );

        // 1C. Unlock X11 pointer (ungrab_pointer + xfixes_show_cursor) in Windowed Mode
        x11_conn
            .ungrab_pointer(smithay::reexports::x11rb::CURRENT_TIME)
            .unwrap();
        x11_conn.xfixes_show_cursor(x11_win).unwrap();
        x11_conn.flush().unwrap();
        for _ in 0..25 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let still_constrained =
            smithay::wayland::pointer_constraints::with_pointer_constraint(&x11_wl_surface, &ptr, |c| {
                c.is_some_and(|c| c.is_active())
            });
        assert!(
            !still_constrained,
            "[{name}] Windowed X11 ungrab_pointer + show_cursor must release the Wayland pointer constraint on x11_wl_surface"
        );

        // Verify post-unlock PointerMotion resumes moving global cursor in Windowed Mode
        let before_unconstrained = comp.pointer_global_location();
        comp.inject_motion(16.0, 12.0, 14.0, 10.0, InputBackendId::Normal);
        let after_unconstrained = comp.pointer_global_location();
        assert!(
            (after_unconstrained.x - (before_unconstrained.x + 16.0)).abs() < 1e-3
                && (after_unconstrained.y - (before_unconstrained.y + 12.0)).abs() < 1e-3,
            "[{name}] Windowed X11 post-unlock PointerMotion must resume moving global cursor"
        );

        // --------------------------------------------------------------------
        // Phase 2: Transition X11 Surface to Fullscreen Mode via _NET_WM_STATE_FULLSCREEN
        //          ClientMessage (XwmHandler::fullscreen_request -> Shell::fullscreen_request)
        // --------------------------------------------------------------------
        let fs_msg = ClientMessageEvent {
            response_type: CLIENT_MESSAGE_EVENT,
            format: 32,
            sequence: 0,
            window: x11_win,
            type_: net_wm_state_atom,
            data: ClientMessageData::from([
                1u32, // _NET_WM_STATE_ADD
                net_wm_state_fullscreen_atom,
                0,
                1,
                0,
            ]),
        };
        x11_conn
            .send_event(
                false,
                root,
                EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                fs_msg,
            )
            .unwrap();
        x11_conn.flush().unwrap();

        for _ in 0..50 {
            let _ = x11_conn.clear_area(true, x11_win, 0, 0, out_w as u16, out_h as u16);
            let _ = x11_conn.flush();
            comp.dispatch();
            if x11_surface.is_fullscreen() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            x11_surface.is_fullscreen(),
            "[{name}] X11 surface must transition to Fullscreen via _NET_WM_STATE_FULLSCREEN ClientMessage"
        );

        // Wait for FULLSCREEN_ANIMATION_DURATION (200ms) and update animations
        std::thread::sleep(Duration::from_millis(220));
        {
            let mut shell = comp.state.common.shell.write();
            shell.update_animations();
        }
        for _ in 0..30 {
            let _ = x11_conn.clear_area(true, x11_win, 0, 0, out_w as u16, out_h as u16);
            let _ = x11_conn.flush();
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }
        while let Ok(Some(_)) = x11_conn.poll_for_event() {}

        // Verify Fullscreen X11Surface has NO 36px SSD headerbar and hits directly at output center!
        let center_pt = Point::from(((out_w as f64) / 2.0, (out_h as f64) / 2.0));
        let hit_fs_x11 = comp
            .surface_under(center_pt, &output)
            .expect("Fullscreen X11 surface must be hit after animation finishes");
        assert_eq!(
            hit_fs_x11.0,
            PointerFocusTarget::X11Surface {
                surface: x11_surface.clone(),
                toplevel: Some(mapped_x11.active_window()),
            },
            "[{name}] Fullscreen hit target must be PointerFocusTarget::X11Surface"
        );

        // 2A. Unlocked PointerMotionAbsolute & PointerMotion in Fullscreen Mode
        comp.inject_motion_absolute(0.5, 0.5, InputBackendId::Normal);
        comp.inject_motion(25.0, -15.0, 20.0, -12.0, InputBackendId::Normal);
        for _ in 0..20 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let pos = comp.pointer_global_location();
        assert!(
            (pos.x - (center_pt.x + 25.0)).abs() < 1e-2
                && (pos.y - (center_pt.y - 15.0)).abs() < 1e-2,
            "[{name}] Fullscreen X11 unlocked motion must advance global cursor"
        );

        let mut fs_motion_events = Vec::new();
        while let Ok(Some(evt)) = x11_conn.poll_for_event() {
            if let smithay::reexports::x11rb::protocol::Event::MotionNotify(m) = evt
                && m.event == x11_win
            {
                fs_motion_events.push((m.event_x, m.event_y, m.root_x, m.root_y));
            }
        }
        assert!(
            !fs_motion_events.is_empty(),
            "[{name}] Fullscreen X11 client must receive MotionNotify events when unlocked"
        );

        // 2B. Pointer Lock in Fullscreen Mode on X11Surface
        x11_conn.xfixes_hide_cursor(x11_win).unwrap();
        x11_conn
            .grab_pointer(
                false,
                x11_win,
                EventMask::POINTER_MOTION,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
                x11_win,
                smithay::reexports::x11rb::NONE,
                smithay::reexports::x11rb::CURRENT_TIME,
            )
            .unwrap()
            .reply()
            .unwrap();
        x11_conn
            .warp_pointer(
                smithay::reexports::x11rb::NONE,
                x11_win,
                0,
                0,
                0,
                0,
                200,
                200,
            )
            .unwrap();
        x11_conn.flush().unwrap();
        for _ in 0..25 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let has_fs_xw_constraint =
            smithay::wayland::pointer_constraints::with_pointer_constraint(&x11_wl_surface, &ptr, |c| {
                c.is_some_and(|c| c.is_active())
            });
        assert!(
            has_fs_xw_constraint,
            "[{name}] Fullscreen X11 grab_pointer + hide_cursor + warp_pointer must activate Wayland pointer constraint on x11_wl_surface"
        );

        let is_fs_locked = smithay::wayland::pointer_constraints::with_pointer_constraint(
            &x11_wl_surface,
            &ptr,
            |c| matches!(c.as_deref(), Some(PointerConstraint::Locked(_))),
        );
        assert!(
            is_fs_locked,
            "[{name}] Fullscreen X11 hide_cursor + grab_pointer + warp_pointer must create PointerConstraint::Locked on x11_wl_surface"
        );
        let fs_lock_origin = comp.pointer_global_location();
        while let Ok(Some(_)) = x11_conn.poll_for_event() {}

        comp.inject_motion(120.0, -90.0, 100.0, -75.0, InputBackendId::Normal);
        comp.inject_motion(-200.0, 140.0, -170.0, 115.0, InputBackendId::Normal);
        for _ in 0..20 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let after_fs_lock_motion = comp.pointer_global_location();
        assert!(
            (after_fs_lock_motion.x - fs_lock_origin.x).abs() < 1e-3
                && (after_fs_lock_motion.y - fs_lock_origin.y).abs() < 1e-3,
            "[{name}] Fullscreen X11 Locked pointer must freeze global cursor location in cosmic-comp"
        );

        let mut locked_fs_x11_motions = Vec::new();
        while let Ok(Some(evt)) = x11_conn.poll_for_event() {
            if let smithay::reexports::x11rb::protocol::Event::MotionNotify(m) = evt
                && m.event == x11_win
            {
                locked_fs_x11_motions.push((m.event_x, m.event_y));
            }
        }
        assert!(
            !locked_fs_x11_motions.is_empty(),
            "[{name}] Fullscreen X11 client must still receive relative-motion-driven events via zwp_relative_pointer_v1 while Locked"
        );

        // 2C. Pointer Unlock in Fullscreen Mode on X11Surface
        x11_conn
            .ungrab_pointer(smithay::reexports::x11rb::CURRENT_TIME)
            .unwrap();
        x11_conn.xfixes_show_cursor(x11_win).unwrap();
        x11_conn.flush().unwrap();
        for _ in 0..25 {
            comp.dispatch();
            std::thread::sleep(Duration::from_millis(5));
        }

        let still_fs_constrained =
            smithay::wayland::pointer_constraints::with_pointer_constraint(&x11_wl_surface, &ptr, |c| {
                c.is_some_and(|c| c.is_active())
            });
        assert!(
            !still_fs_constrained,
            "[{name}] Fullscreen X11 ungrab_pointer + show_cursor must release Wayland pointer constraint"
        );

        let before_fs_unconstrained = comp.pointer_global_location();
        comp.inject_motion(-22.0, 18.0, -18.0, 15.0, InputBackendId::Normal);
        let after_fs_unconstrained = comp.pointer_global_location();
        assert!(
            (after_fs_unconstrained.x - (before_fs_unconstrained.x - 22.0)).abs() < 1e-3
                && (after_fs_unconstrained.y - (before_fs_unconstrained.y + 18.0)).abs() < 1e-3,
            "[{name}] Fullscreen X11 post-unlock PointerMotion must resume moving global cursor"
        );

        // --------------------------------------------------------------------
        // Phase 3: Unfullscreen X11 Surface via _NET_WM_STATE_REMOVE ClientMessage
        //          back to Windowed Mode for the next resolution iteration
        // --------------------------------------------------------------------
        let unfs_msg = ClientMessageEvent {
            response_type: CLIENT_MESSAGE_EVENT,
            format: 32,
            sequence: 0,
            window: x11_win,
            type_: net_wm_state_atom,
            data: ClientMessageData::from([
                0u32, // _NET_WM_STATE_REMOVE
                net_wm_state_fullscreen_atom,
                0,
                1,
                0,
            ]),
        };
        x11_conn
            .send_event(
                false,
                root,
                EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                unfs_msg,
            )
            .unwrap();
        x11_conn.flush().unwrap();

        for _ in 0..50 {
            let _ = x11_conn.clear_area(true, x11_win, 0, 0, 600, 400);
            let _ = x11_conn.flush();
            comp.dispatch();
            if !x11_surface.is_fullscreen() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !x11_surface.is_fullscreen(),
            "[{name}] X11 surface must return to Windowed mode after _NET_WM_STATE_REMOVE"
        );
    }
}

