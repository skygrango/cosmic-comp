// SPDX-License-Identifier: GPL-3.0-only

use crate::{backend::kms::drm_helpers::HdrOutputState, state::State, utils::prelude::SeatExt};
use smithay::{
    desktop::utils::surface_primary_scanout_output,
    output::Output,
    reexports::{
        wayland_protocols::wp::color_management::v1::server::wp_image_description_info_v1::WpImageDescriptionInfoV1,
        wayland_server::protocol::wl_surface::WlSurface,
    },
    wayland::{
        color::{
            management::{
                ColorManagementHandler, ColorManagementState, ImageDescription, Primaries,
                PrimariesOption, TransferFunction, send_image_description_info,
            },
            representation::{ColorRepresentationHandler, ColorRepresentationState},
        },
        compositor::{get_parent, with_states},
    },
};

pub(crate) fn description_for_output(output: &Output) -> ImageDescription {
    let Some(active) = output
        .user_data()
        .get::<HdrOutputState>()
        .and_then(|s| s.get().or_else(|| s.staged()))
    else {
        return ImageDescription::SRGB;
    };

    let caps = active.capabilities;
    let max_luminance = if caps.max_luminance > 0 {
        u32::from(caps.max_luminance)
    } else {
        1_000
    };
    let min_luminance = u32::from(caps.min_luminance);
    let reference_white = if active.reference_white > 0 {
        u32::from(active.reference_white)
    } else {
        203
    };

    ImageDescription {
        transfer: TransferFunction::St2084Pq,
        primaries: PrimariesOption {
            named: Some(Primaries::Bt2020),
            values: None,
        },
        max_cll: (caps.max_luminance > 0).then(|| u32::from(caps.max_luminance)),
        max_fall: (caps.max_frame_average_luminance > 0)
            .then(|| u32::from(caps.max_frame_average_luminance)),
        mastering_luminance: None,
        mastering_primaries: None,
        luminances: Some((min_luminance, max_luminance, reference_white)),
        windows_scrgb: false,
        windows_bt2100: false,
    }
}

impl ColorManagementHandler for State {
    fn color_management_state(&mut self) -> &mut ColorManagementState {
        &mut self.common.color_management_state
    }

    fn description_for_output(&mut self, output: &Output) -> ImageDescription {
        description_for_output(output)
    }

    fn preferred_description_for_surface(&mut self, surface: &WlSurface) -> ImageDescription {
        let mut root = surface.clone();
        while let Some(parent) = get_parent(&root) {
            root = parent;
        }

        let shell = self.common.shell.read();
        let output = with_states(&root, |states| {
            surface_primary_scanout_output(&root, states)
        })
        .or_else(|| shell.visible_output_for_surface(&root).cloned())
        .or_else(|| {
            // Fallback to active/focused output so surfaces created before being mapped
            // receive the correct HDR preference during startup probing.
            Some(shell.seats.last_active().focused_or_active_output())
        })
        .or_else(|| shell.outputs().next().cloned());

        output
            .as_ref()
            .map(description_for_output)
            .unwrap_or(ImageDescription::SRGB)
    }

    fn schedule_image_description_info(
        &mut self,
        info: WpImageDescriptionInfoV1,
        desc: ImageDescription,
    ) {
        self.common.event_loop_handle.insert_idle(move |_state| {
            send_image_description_info(&info, &desc);
        });
    }
}

impl ColorRepresentationHandler for State {
    fn color_representation_state(&mut self) -> &mut ColorRepresentationState {
        &mut self.common.color_representation_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::kms::drm_helpers::{ActiveHdrOutput, HdrSinkCapabilities};
    use smithay::output::{PhysicalProperties, Subpixel};

    #[test]
    fn reports_the_hardware_validated_hdr_description() {
        let output = Output::new(
            "DP-test".into(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: "Test".into(),
                model: "HDR".into(),
                serial_number: "test".into(),
            },
        );
        output
            .user_data()
            .insert_if_missing_threadsafe(HdrOutputState::default);
        output
            .user_data()
            .get::<HdrOutputState>()
            .unwrap()
            .set(Some(ActiveHdrOutput {
                capabilities: HdrSinkCapabilities {
                    max_luminance: 993,
                    min_luminance: 10,
                    max_frame_average_luminance: 993,
                },
                native_primaries: None,
                reference_white: 203,
            }));

        let desc = description_for_output(&output);
        assert_eq!(desc.transfer, TransferFunction::St2084Pq);
        assert_eq!(desc.primaries.named, Some(Primaries::Bt2020));
        assert_eq!(desc.luminances, Some((10, 993, 203)));
        assert_eq!(desc.max_cll, Some(993));
        assert_eq!(desc.max_fall, Some(993));
    }
}
