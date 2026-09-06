use smithay::{
    backend::renderer::{
        ImportAll, Renderer,
        element::surface::{KindEvaluation, WaylandSurfaceRenderElement},
        utils::RendererSurfaceStateUserData,
    },
    reexports::wayland_server::protocol::wl_surface,
    render_elements,
    utils::{Logical, Physical, Point, Rectangle, Scale},
    wayland::{
        color::management::ImageDescription,
        compositor::{self, TraversalAction},
    },
};
use tracing::warn;

use crate::backend::render::{
    element::AsGlowRenderer,
    wayland::{blur_effect::BlurElement, clipped_surface::ClippedSurfaceRenderElement},
};

pub mod blur_effect;
pub mod clipped_surface;

render_elements! {
    pub SurfaceRenderElement<R> where R: AsGlowRenderer + ImportAll, R::TextureId: Send;
    Blur=BlurElement,
    Clipped=ClippedSurfaceRenderElement<R>,
    Wayland=WaylandSurfaceRenderElement<R>,
}

pub fn push_render_elements_from_surface_tree<R>(
    renderer: &mut R,
    main_surface: &wl_surface::WlSurface,
    location: impl Into<Point<i32, Physical>>,
    geometry: impl Into<Rectangle<f64, Logical>>,
    scale: impl Into<Scale<f64>>,
    alpha: f32,
    should_clip: bool,
    radii: [u8; 4],
    blur_geometry: impl Into<Option<Rectangle<f64, Logical>>>,
    blur_strength: usize,
    kind: impl Into<KindEvaluation>,
    push_above: &mut dyn FnMut(SurfaceRenderElement<R>),
    mut push_below: Option<&mut dyn FnMut(SurfaceRenderElement<R>)>,
) where
    R: Renderer + ImportAll + AsGlowRenderer,
    R::TextureId: Clone + 'static,
{
    let location = location.into().to_f64();
    let geometry = geometry.into();
    let blur_geometry = blur_geometry.into();
    let scale = scale.into();
    let kind = kind.into();
    let mut passed_main = false;

    compositor::with_surface_tree_downward(
        main_surface,
        location,
        |_, states, location| {
            let mut location = *location;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();

            if let Some(data) = data {
                if let Some(view) = data.lock().unwrap().view() {
                    location += view.offset.to_f64().to_physical(scale);
                    TraversalAction::DoChildren(location)
                } else {
                    TraversalAction::SkipChildren
                }
            } else {
                TraversalAction::SkipChildren
            }
        },
        |surface, states, location| {
            let mut location = *location;
            let kind = kind.eval(states);
            let data = states.data_map.get::<RendererSurfaceStateUserData>();
            let mut blur = Ok(None);

            if let Some(data) = data {
                let has_view = if let Some(view) = data.lock().unwrap().view() {
                    location += view.offset.to_f64().to_physical(scale);

                    true
                } else {
                    false
                };

                if has_view {
                    match WaylandSurfaceRenderElement::from_surface(
                        renderer, surface, states, location, alpha, kind,
                    ) {
                        Ok(Some(element)) => {
                            let blur_geo = blur_geometry.unwrap_or(geometry);
                            blur = BlurElement::from_surface(
                                renderer,
                                states,
                                blur_geo,
                                scale.x,
                                radii,
                                blur_strength,
                            );
                            let elem: SurfaceRenderElement<R> = if radii.iter().any(|r| *r != 0)
                                && should_clip
                                && ClippedSurfaceRenderElement::will_clip(
                                    &element, scale, geometry, radii,
                                ) {
                                ClippedSurfaceRenderElement::new(
                                    renderer, element, scale, geometry, radii,
                                )
                                .into()
                            } else {
                                element.into()
                            };
                            if let Some(push_below) = push_below.as_mut()
                                && passed_main
                            {
                                push_below(elem);
                            } else {
                                push_above(elem);
                            }
                        }
                        Ok(None) => {} // surface is not mapped
                        Err(err) => {
                            warn!("Failed to import surface: {:?}", err);
                        }
                    }
                }
            }

            if surface == main_surface {
                passed_main = true;
            }

            if let Ok(Some(elem)) = blur {
                if let Some(push_below) = push_below.as_mut()
                    && passed_main
                {
                    push_below(elem.into());
                } else {
                    push_above(elem.into());
                }
            }
        },
        |_, _, _| true,
    );
}

/// Rescales an output's reference white for Windows-scRGB content: the
/// encoding's 1.0 is its `max` luminance (80 cd/m²) and its SDR white is the
/// `reference` (203 cd/m² per BT.2408), so mapping that reference onto the
/// output's reference keeps SDR-in-scRGB at the same brightness as native SDR.
#[cfg_attr(not(test), allow(dead_code))]
fn scrgb_reference_scale(description: &ImageDescription) -> f32 {
    description
        .luminances
        .map(|(_min, max, reference)| {
            if reference == 0 {
                1.0
            } else {
                max as f32 / reference as f32
            }
        })
        .unwrap_or(80.0 / 203.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrgb_reference_scale_matches_windows_conventions() {
        let scale = scrgb_reference_scale(&ImageDescription::WINDOWS_SCRGB);
        assert!((scale - 80.0 / 203.0).abs() < 1e-6);
    }
}
