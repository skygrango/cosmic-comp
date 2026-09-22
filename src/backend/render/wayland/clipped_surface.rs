// Taken and modified from niri, licensed GPL-3.

use std::borrow::{Borrow, BorrowMut};

use glam::{Affine2, Mat3, Vec2};
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};
use smithay::{
    backend::renderer::{
        Frame, ImportAll, Renderer,
        element::{
            Element, Id, Kind, RenderElement, UnderlyingStorage,
            surface::WaylandSurfaceRenderElement,
        },
        gles::{GlesFrame, GlesRenderer, GlesTexProgram, Uniform, UniformValue},
        utils::{CommitCounter, DamageSet, OpaqueRegions},
    },
    utils::user_data::UserDataMap,
};

use crate::backend::render::element::AsGlowRenderer;

pub static CLIPPING_SHADER: &str = include_str!("../shaders/clipped_surface.frag");
pub struct ClippingShader(pub GlesTexProgram);

impl ClippingShader {
    pub fn get<R: AsGlowRenderer>(renderer: &R) -> GlesTexProgram {
        let Some(glow) = renderer.glow_renderer() else {
            panic!("ClippingShader requires a GlowRenderer");
        };
        Borrow::<GlesRenderer>::borrow(glow)
            .egl_context()
            .user_data()
            .get::<ClippingShader>()
            .expect("Custom Shaders not initialized")
            .0
            .clone()
    }
}

#[derive(Debug)]
pub struct ClippedSurfaceRenderElement<R: Renderer> {
    inner: WaylandSurfaceRenderElement<R>,
    program: Option<GlesTexProgram>,
    radius: [u8; 4],
    geometry: Rectangle<f64, Logical>,
    scale: Scale<f64>,
    physical_geo: Rectangle<i32, Physical>,
    physical_radii: [f32; 4],
    physical_corners: [Rectangle<i32, Physical>; 4],
    uniforms: Vec<Uniform<'static>>,
}

impl<R> ClippedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll,
{
    pub fn new(
        renderer: &mut R,
        elem: WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> Self
    where
        R: AsGlowRenderer,
    {
        let physical_geo = geometry.to_physical_precise_round(scale);
        let physical_corners = Self::physical_corners(geometry, radius, scale);
        Self::new_precomputed(
            renderer,
            elem,
            scale,
            geometry,
            radius,
            physical_geo,
            physical_corners,
        )
    }

    pub fn new_precomputed(
        renderer: &mut R,
        elem: WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
        physical_geo: Rectangle<i32, Physical>,
        physical_corners: [Rectangle<i32, Physical>; 4],
    ) -> Self
    where
        R: AsGlowRenderer,
    {
        let physical_radii = [
            radius[0] as f32 * scale.x as f32,
            radius[1] as f32 * scale.y as f32,
            radius[2] as f32 * scale.x as f32,
            radius[3] as f32 * scale.y as f32,
        ];

        let (program, uniforms) = if renderer.glow_renderer().is_some() {
            let elem_geo = elem.geometry(scale);
            let geo = physical_geo;
            let buf_size = elem.buffer_size();
            let view = elem.view();

            let transform = elem.transform();
            let transform_matrix = Affine2::from_translation(Vec2::new(0.5, 0.5))
                * transform.matrix()
                * Affine2::from_translation(-Vec2::new(0.5, 0.5));

            let geo_scale = {
                let Scale { x, y } = elem_geo.size.to_f64() / geo.size.to_f64();
                Affine2::from_scale(Vec2::new(x as f32, y as f32))
            };

            let geo_translation = {
                let offset = (elem_geo.loc - geo.loc).to_f64();
                Affine2::from_translation(Vec2::new(
                    (offset.x / elem_geo.size.w as f64) as f32,
                    (offset.y / elem_geo.size.h as f64) as f32,
                ))
            };

            let buf_scale = {
                let Scale { x, y } = buf_size.to_f64() / view.src.size.to_f64();
                Affine2::from_scale(Vec2::new(x as f32, y as f32))
            };

            let buf_translation = Affine2::from_translation(Vec2::new(
                (view.src.loc.x / buf_size.w as f64) as f32,
                (view.src.loc.y / buf_size.h as f64) as f32,
            ));

            let input_to_geo = Mat3::from(
                transform_matrix * geo_scale * geo_translation * buf_scale * buf_translation,
            );

            let hdr_config = renderer
                .glow_renderer()
                .and_then(|glow| Borrow::<GlesRenderer>::borrow(glow).hdr_output());
            let (hdr_enabled, ref_white, sdr_gamma, gamut_stretch, hw_offload, is_sdr, max_lum) =
                if let Some(config) = hdr_config {
                    (
                        1.0_f32,
                        config.reference_white,
                        config.sdr_gamma,
                        config.gamut_stretch,
                        if config.hardware_offload {
                            1.0_f32
                        } else {
                            0.0_f32
                        },
                        if config.is_sdr { 1.0_f32 } else { 0.0_f32 },
                        config.max_luminance,
                    )
                } else {
                    (
                        0.0_f32, 203.0_f32, 2.2_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1000.0_f32,
                    )
                };

            let uniforms = vec![
                Uniform::new("geo_size", (geometry.size.w as f32, geometry.size.h as f32)),
                Uniform::new(
                    "corner_radius",
                    [
                        radius[0] as f32,
                        radius[1] as f32,
                        radius[2] as f32,
                        radius[3] as f32,
                    ],
                ),
                Uniform::new(
                    "input_to_geo",
                    UniformValue::Matrix3x3 {
                        matrices: vec![*AsRef::<[f32; 9]>::as_ref(&input_to_geo)],
                        transpose: false,
                    },
                ),
                Uniform::new("noise", UniformValue::_1f(0.0)),
                Uniform::new("hdr_enabled", UniformValue::_1f(hdr_enabled)),
                Uniform::new("hdr_reference_white", UniformValue::_1f(ref_white)),
                Uniform::new("hdr_sdr_gamma", UniformValue::_1f(sdr_gamma)),
                Uniform::new("hdr_gamut_stretch", UniformValue::_1f(gamut_stretch)),
                Uniform::new("hdr_hardware_offload", UniformValue::_1f(hw_offload)),
                Uniform::new("hdr_target_is_sdr", UniformValue::_1f(is_sdr)),
                Uniform::new("hdr_input_pq", UniformValue::_1f(0.0)),
                Uniform::new("hdr_input_hlg", UniformValue::_1f(0.0)),
                Uniform::new("hdr_input_primaries", UniformValue::_1f(0.0)),
                Uniform::new("hdr_content_reference", UniformValue::_1f(203.0)),
                Uniform::new("hdr_max_content_luminance", UniformValue::_1f(1000.0)),
                Uniform::new("hdr_max_destination_luminance", UniformValue::_1f(max_lum)),
            ];
            (Some(ClippingShader::get(renderer)), uniforms)
        } else {
            (None, Vec::new())
        };

        Self {
            inner: elem,
            program,
            radius,
            geometry,
            scale,
            physical_geo,
            physical_radii,
            physical_corners,
            uniforms,
        }
    }

    pub fn physical_corners(
        geo: Rectangle<f64, Logical>,
        radius: [u8; 4],
        scale: Scale<f64>,
    ) -> [Rectangle<i32, Physical>; 4] {
        let corners = Self::rounded_corners(geo, radius);
        corners.map(|rect| rect.to_physical_precise_up(scale))
    }

    #[inline]
    pub fn will_clip_precomputed(
        elem: &WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        physical_geo: Rectangle<i32, Physical>,
        physical_corners: &[Rectangle<i32, Physical>; 4],
    ) -> bool {
        let elem_geo = elem.geometry(scale);

        // If elem_geo extends outside geometry, it needs clipping.
        if !physical_geo.contains_rect(elem_geo) {
            return true;
        }

        // If elem_geo is inside geo, it needs clipping if and only if it overlaps any non-zero rounded corner.
        physical_corners
            .iter()
            .any(|c| !c.is_empty() && c.overlaps(elem_geo))
    }

    pub fn will_clip(
        elem: &WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> bool {
        let physical_geo = geometry.to_physical_precise_round(scale);
        let physical_corners = Self::physical_corners(geometry, radius, scale);
        Self::will_clip_precomputed(elem, scale, physical_geo, &physical_corners)
    }

    fn rounded_corners(
        geo: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> [Rectangle<f64, Logical>; 4] {
        let top_left = radius[0] as f64;
        let top_right = radius[1] as f64;
        let bottom_right = radius[2] as f64;
        let bottom_left = radius[3] as f64;

        [
            Rectangle::new(geo.loc, Size::from((top_left, top_left))),
            Rectangle::new(
                Point::from((geo.loc.x + geo.size.w - top_right, geo.loc.y)),
                Size::from((top_right, top_right)),
            ),
            Rectangle::new(
                Point::from((
                    geo.loc.x + geo.size.w - bottom_right,
                    geo.loc.y + geo.size.h - bottom_right,
                )),
                Size::from((bottom_right, bottom_right)),
            ),
            Rectangle::new(
                Point::from((geo.loc.x, geo.loc.y + geo.size.h - bottom_left)),
                Size::from((bottom_left, bottom_left)),
            ),
        ]
    }

    pub fn recover_uncropped_dst(
        element_src: Rectangle<f64, Buffer>,
        transform: Transform,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
    ) -> Rectangle<i32, Physical> {
        if src != element_src && src.size.w > 0.0 && src.size.h > 0.0 {
            let unscaled_physical_crop = transform.invert().transform_size(dst.size).to_f64();
            if unscaled_physical_crop.w > 0.0 && unscaled_physical_crop.h > 0.0 {
                let physical_to_buffer_scale = Scale::from((
                    src.size.w / unscaled_physical_crop.w,
                    src.size.h / unscaled_physical_crop.h,
                ));
                let mut relative_src = src;
                relative_src.loc -= element_src.loc;
                let relative_logical =
                    relative_src.to_logical(physical_to_buffer_scale, transform, &element_src.size);
                let uncropped_size_logical = element_src
                    .size
                    .to_logical(physical_to_buffer_scale, transform);
                let uncropped_size = transform.transform_size(uncropped_size_logical);
                let uncropped_loc = Point::from((
                    dst.loc.x - relative_logical.loc.x.round() as i32,
                    dst.loc.y - relative_logical.loc.y.round() as i32,
                ));

                Rectangle::new(
                    uncropped_loc,
                    Size::from((
                        uncropped_size.w.round() as i32,
                        uncropped_size.h.round() as i32,
                    )),
                )
            } else {
                dst
            }
        } else {
            dst
        }
    }

    pub fn clip_for_uncropped(
        physical_geo: Rectangle<i32, Physical>,
        physical_radii: [f32; 4],
        inner_geo: Rectangle<i32, Physical>,
        uncropped_dst: Rectangle<i32, Physical>,
    ) -> (Rectangle<i32, Physical>, [f32; 4]) {
        if uncropped_dst == physical_geo || uncropped_dst == inner_geo {
            (physical_geo, physical_radii)
        } else if inner_geo.size.w > 0 && inner_geo.size.h > 0 {
            let sx = uncropped_dst.size.w as f64 / inner_geo.size.w as f64;
            let sy = uncropped_dst.size.h as f64 / inner_geo.size.h as f64;
            let ox = (physical_geo.loc.x - inner_geo.loc.x) as f64;
            let oy = (physical_geo.loc.y - inner_geo.loc.y) as f64;

            let clip_loc = Point::from((
                uncropped_dst.loc.x + (ox * sx).round() as i32,
                uncropped_dst.loc.y + (oy * sy).round() as i32,
            ));
            let clip_size = Size::from((
                (physical_geo.size.w as f64 * sx).round() as i32,
                (physical_geo.size.h as f64 * sy).round() as i32,
            ));
            let s_radius = (sx.min(sy)) as f32;
            let clip_radii = [
                physical_radii[0] * s_radius,
                physical_radii[1] * s_radius,
                physical_radii[2] * s_radius,
                physical_radii[3] * s_radius,
            ];
            (Rectangle::new(clip_loc, clip_size), clip_radii)
        } else {
            (physical_geo, physical_radii)
        }
    }

    pub fn clip_for_draw(
        &self,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
    ) -> (Rectangle<i32, Physical>, [f32; 4]) {
        let uncropped_dst =
            Self::recover_uncropped_dst(self.inner.src(), self.inner.transform(), src, dst);
        Self::clip_for_uncropped(
            self.physical_geo,
            self.physical_radii,
            self.inner.geometry(self.scale),
            uncropped_dst,
        )
    }

    #[inline]
    pub fn clip_for_dst(
        &self,
        dst: Rectangle<i32, Physical>,
    ) -> (Rectangle<i32, Physical>, [f32; 4]) {
        self.clip_for_draw(self.inner.src(), dst)
    }
}

impl<R> Element for ClippedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + AsGlowRenderer,
    R::TextureId: 'static,
{
    fn id(&self) -> &Id {
        self.inner.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
    }

    fn transform(&self) -> Transform {
        self.inner.transform()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        // FIXME: radius changes need to cause damage.
        let damage = self.inner.damage_since(scale, commit);

        // Intersect with geometry, since we're clipping by it.
        let mut geo = if scale == self.scale {
            self.physical_geo
        } else {
            self.geometry.to_physical_precise_round(scale)
        };
        geo.loc -= self.geometry(scale).loc;
        damage
            .into_iter()
            .filter_map(|rect| rect.intersection(geo))
            .collect()
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        let regions = self.inner.opaque_regions(scale);

        // Intersect with geometry, since we're clipping by it.
        let mut geo = if scale == self.scale {
            self.physical_geo
        } else {
            self.geometry.to_physical_precise_round(scale)
        };
        geo.loc -= self.geometry(scale).loc;
        let regions = regions
            .into_iter()
            .filter_map(|rect| rect.intersection(geo));

        // Subtract the rounded corners.
        let corners = if scale == self.scale {
            self.physical_corners
        } else {
            Self::physical_corners(self.geometry, self.radius, scale)
        };

        let elem_loc = self.geometry(scale).loc;
        let corners = corners.into_iter().map(|mut rect| {
            rect.loc -= elem_loc;
            rect
        });

        OpaqueRegions::from_slice(&Rectangle::subtract_rects_many(regions, corners))
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

impl<R> RenderElement<R> for ClippedSurfaceRenderElement<R>
where
    R: AsGlowRenderer + Renderer + ImportAll,
    R::TextureId: 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        frame.set_surface_clip(Some(self.clip_for_draw(src, dst)));

        let previous_override =
            <R as AsGlowRenderer>::glow_frame_mut(frame).and_then(|glow_frame| {
                let gles_frame = BorrowMut::<GlesFrame>::borrow_mut(glow_frame);
                let previous = gles_frame.take_tex_program_override();
                if let Some(ref program) = self.program {
                    gles_frame.override_default_tex_program(program.clone(), self.uniforms.clone());
                }
                previous
            });
        let res = self
            .inner
            .draw(frame, src, dst, damage, opaque_regions, cache);
        if let Some(glow_frame) = <R as AsGlowRenderer>::glow_frame_mut(frame) {
            BorrowMut::<GlesFrame>::borrow_mut(glow_frame)
                .set_tex_program_override(previous_override);
        }
        frame.set_surface_clip(None);
        res?;
        Ok(())
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_corners_calculation() {
        let geo = Rectangle::new(Point::from((100.0, 200.0)), Size::from((800.0, 600.0)));
        let radii = [16, 16, 16, 16];
        let corners = ClippedSurfaceRenderElement::<smithay::backend::renderer::glow::GlowRenderer>::rounded_corners(geo, radii);

        // Top-left
        assert_eq!(corners[0].loc, Point::from((100.0, 200.0)));
        assert_eq!(corners[0].size, Size::from((16.0, 16.0)));

        // Top-right
        assert_eq!(corners[1].loc, Point::from((100.0 + 800.0 - 16.0, 200.0)));
        assert_eq!(corners[1].size, Size::from((16.0, 16.0)));

        // Bottom-right
        assert_eq!(
            corners[2].loc,
            Point::from((100.0 + 800.0 - 16.0, 200.0 + 600.0 - 16.0))
        );
        assert_eq!(corners[2].size, Size::from((16.0, 16.0)));

        // Bottom-left
        assert_eq!(corners[3].loc, Point::from((100.0, 200.0 + 600.0 - 16.0)));
        assert_eq!(corners[3].size, Size::from((16.0, 16.0)));
    }

    #[test]
    fn test_physical_corners_and_clipping() {
        let geo = Rectangle::new(Point::from((100.0, 200.0)), Size::from((800.0, 600.0)));
        let radii = [16, 16, 16, 16];
        let scale = Scale::from(2.0);
        let physical_corners = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::physical_corners(geo, radii, scale);
        let physical_geo = geo.to_physical_precise_round(scale);

        assert_eq!(physical_geo.loc, Point::from((200, 400)));
        assert_eq!(physical_geo.size, Size::from((1600, 1200)));

        assert_eq!(physical_corners[0].loc, Point::from((200, 400)));
        assert_eq!(physical_corners[0].size, Size::from((32, 32)));

        // Test inner rect that overlaps corner
        let corner_elem_geo = Rectangle::new(Point::from((200, 400)), Size::from((16, 16)));
        assert!(
            physical_corners
                .iter()
                .any(|c| !c.is_empty() && c.overlaps(corner_elem_geo))
        );

        // Test inner rect that is purely in the center (does not overlap any corner)
        let center_elem_geo = Rectangle::new(Point::from((300, 500)), Size::from((100, 100)));
        assert!(
            !physical_corners
                .iter()
                .any(|c| !c.is_empty() && c.overlaps(center_elem_geo))
        );
        assert!(physical_geo.contains_rect(center_elem_geo));

        // Test rect extending outside bounds
        let outside_elem_geo = Rectangle::new(Point::from((150, 400)), Size::from((100, 100)));
        assert!(!physical_geo.contains_rect(outside_elem_geo));
    }

    #[test]
    fn test_recover_uncropped_dst_fractional_scale() {
        // Simulating a window at logical x = -100, y = 100, w = 800, h = 600 at 1.5x scale:
        // Physical uncropped: x = -150, y = 150, w = 1200, h = 900
        // Buffer src: loc (0, 0), size (1200, 900)
        let element_src: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((0.0, 0.0)), Size::from((1200.0, 900.0)));
        let transform = Transform::Normal;

        // 1. Crossing left screen edge at x = 0:
        // Visible physical part on screen: loc (0, 150), size (1050, 900)
        // Buffer src cropped by 150px: loc (150, 0), size (1050, 900)
        let cropped_dst: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((0, 150)), Size::from((1050, 900)));
        let cropped_src: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((150.0, 0.0)), Size::from((1050.0, 900.0)));

        let recovered = ClippedSurfaceRenderElement::<smithay::backend::renderer::glow::GlowRenderer>::recover_uncropped_dst(
            element_src,
            transform,
            cropped_src,
            cropped_dst,
        );

        assert_eq!(recovered.loc, Point::from((-150, 150)));
        assert_eq!(recovered.size, Size::from((1200, 900)));

        // 2. Crossing right screen edge (e.g. at 2560px with window at 2000px):
        // Visible physical part on screen: loc (2000, 150), size (560, 900)
        // Buffer src cropped on right: loc (0, 0), size (560, 900)
        let right_cropped_dst: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((2000, 150)), Size::from((560, 900)));
        let right_cropped_src: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((0.0, 0.0)), Size::from((560.0, 900.0)));

        let recovered_right = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::recover_uncropped_dst(
            element_src, transform, right_cropped_src, right_cropped_dst
        );

        assert_eq!(recovered_right.loc, Point::from((2000, 150)));
        assert_eq!(recovered_right.size, Size::from((1200, 900)));

        // 3. Top and bottom crops:
        let top_cropped_dst: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((150, 0)), Size::from((1200, 800)));
        let top_cropped_src: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((0.0, 100.0)), Size::from((1200.0, 800.0)));

        let recovered_top = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::recover_uncropped_dst(
            element_src, transform, top_cropped_src, top_cropped_dst
        );

        assert_eq!(recovered_top.loc, Point::from((150, -100)));
        assert_eq!(recovered_top.size, Size::from((1200, 900)));

        // 4. 100% scale (unscaled 1.0x):
        let elem_src_100: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((0.0, 0.0)), Size::from((800.0, 600.0)));
        // Crossing left edge: x = -200, w = 800 -> visible dst: [0, 100, 600, 600], src: [200, 0, 600, 600]
        let cropped_dst_100: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((0, 100)), Size::from((600, 600)));
        let cropped_src_100: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((200.0, 0.0)), Size::from((600.0, 600.0)));

        let recovered_100 = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::recover_uncropped_dst(
            elem_src_100, transform, cropped_src_100, cropped_dst_100
        );
        assert_eq!(recovered_100.loc, Point::from((-200, 100)));
        assert_eq!(recovered_100.size, Size::from((800, 600)));
    }

    #[test]
    fn test_clip_for_uncropped_no_false_edge_rounding() {
        let physical_geo: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((-150, 150)), Size::from((1200, 900)));
        let physical_radii = [12.0f32, 12.0f32, 12.0f32, 12.0f32];
        let inner_geo = physical_geo;

        // When recovered uncropped destination matches the true window position:
        let uncropped_dst = physical_geo;

        let (clip_rect, clip_radii) = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::clip_for_uncropped(
            physical_geo, physical_radii, inner_geo, uncropped_dst
        );

        // Clip rectangle remains at the true window position, NOT at the screen edge (x = 0)
        assert_eq!(clip_rect, physical_geo);
        assert_eq!(clip_radii, physical_radii);

        // Test subsurface within the window:
        let subsurface_geo: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((-50, 200)), Size::from((400, 300)));
        let (sub_clip_rect, sub_clip_radii) = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::clip_for_uncropped(
            physical_geo,
            physical_radii,
            subsurface_geo,
            subsurface_geo,
        );
        // Subsurface clip still covers the entire window bounding box:
        assert_eq!(sub_clip_rect, physical_geo);
        assert_eq!(sub_clip_radii, physical_radii);
    }

    #[test]
    fn test_outline_render_element_recovers_uncropped_dst() {
        use crate::backend::render::OutlineRenderElement;
        use smithay::backend::renderer::element::Element;
        use smithay::utils::Transform;

        // Logical area: loc (-200, 100), size (800, 600)
        let area: Rectangle<i32, Logical> =
            Rectangle::new(Point::from((-200, 100)), Size::from((800, 600)));
        let outline = OutlineRenderElement {
            id: smithay::backend::renderer::element::Id::new(),
            commit: smithay::backend::renderer::utils::CommitCounter::default(),
            area,
            thickness: 2.0,
            radius: [16.0, 16.0, 16.0, 16.0],
            color: [1.0, 1.0, 1.0],
            alpha: 1.0,
        };

        // At 1.5x scale, physical uncropped geometry is (-300, 150, 1200, 900)
        let element_src = outline.src();
        assert_eq!(element_src.size, Size::from((800.0, 600.0)));

        // Cropped by left screen boundary (x = 0):
        // Visible physical dst: loc (0, 150), size (900, 900)
        // Relative cropped src: loc (200, 0), size (600, 600)
        let cropped_dst: Rectangle<i32, Physical> =
            Rectangle::new(Point::from((0, 150)), Size::from((900, 900)));
        let cropped_src: Rectangle<f64, Buffer> =
            Rectangle::new(Point::from((200.0, 0.0)), Size::from((600.0, 600.0)));

        let recovered_dst = ClippedSurfaceRenderElement::<
            smithay::backend::renderer::glow::GlowRenderer,
        >::recover_uncropped_dst(
            element_src, Transform::Normal, cropped_src, cropped_dst
        );

        assert_eq!(recovered_dst.loc, Point::from((-300, 150)));
        assert_eq!(recovered_dst.size, Size::from((1200, 900)));

        // Verify invariant scale calculation:
        let scale = recovered_dst.size.w as f32 / area.size.w as f32;
        assert_eq!(scale, 1.5);
        assert_eq!(outline.thickness * scale, 3.0);
        assert_eq!(outline.radius.map(|r| r * scale), [24.0, 24.0, 24.0, 24.0]);
    }
}
