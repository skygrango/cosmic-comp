use std::{any::Any, borrow::Borrow, cell::RefCell, collections::HashMap};

use glam::{Affine2, Mat3, Vec2};
use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            ErasedContextId, Frame, ImportAll, ImportMem, Renderer, Texture,
            element::{Element, Id, Kind, RenderElement, UnderlyingStorage},
            gles::{
                GlesPixelProgram, GlesRenderer, Uniform, UniformValue, element::PixelShaderElement,
            },
            glow::GlowRenderer,
            utils::{CommitCounter, DamageSet, OpaqueRegions},
        },
    },
    utils::{
        Buffer as BufferCoords, IsAlive, Logical, Physical, Point, Rectangle, Scale, Size,
        Transform, user_data::UserDataMap,
    },
};

use crate::{
    backend::render::element::AsGlowRenderer,
    shell::element::CosmicMappedKey,
    utils::prelude::{Local, RectLocalExt},
};

pub static SHADOW_SHADER: &str = include_str!("./shaders/shadow.frag");
pub struct ShadowShader(pub GlesPixelProgram);

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowParameters {
    geo: Rectangle<i32, Local>,
    scale: f64,
    alpha: f32,
    radius: [u8; 4],
    dark_mode: bool,
}
type ShadowCache = RefCell<HashMap<CosmicMappedKey, (ShadowParameters, PixelShaderElement)>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NinePatchKey {
    context_id: ErasedContextId,
    radius: [u8; 4],
    scale_bits: u64,
    dark_mode: bool,
}

struct CachedNinePatchTexture {
    texture: Box<dyn Any + Send + 'static>,
    size: Size<i32, BufferCoords>,
    margins: (i32, i32, i32, i32),
}

struct CachedNinePatchWindow {
    params: ShadowParameters,
    id: Id,
    commit: CommitCounter,
}

static FORCE_NINEPATCH: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("COSMIC_FORCE_NINEPATCH_SHADOW").is_some());

thread_local! {
    static NINE_PATCH_CACHE: RefCell<HashMap<NinePatchKey, CachedNinePatchTexture>> = RefCell::new(HashMap::new());
    static NINE_PATCH_ELEMENT_CACHE: RefCell<HashMap<CosmicMappedKey, CachedNinePatchWindow>> = RefCell::new(HashMap::new());
}

// CC0 Gaussian and erf approximation functions taken from Evan Wallace:
// https://madebyevan.com/shaders/fast-rounded-rectangle-shadows/
fn erf_scalar(x: f32) -> f32 {
    let s = x.signum();
    let a = x.abs();
    let p = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    let p2 = p * p;
    let p4 = p2 * p2;
    s - s / p4
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    use std::f32::consts::PI;
    (-(x * x) / (2.0 * sigma * sigma)).exp() / ((2.0 * PI).sqrt() * sigma)
}

fn rounded_box_shadow_x(x: f32, y: f32, sigma: f32, corner: f32, half_w: f32, half_h: f32) -> f32 {
    let delta = (half_h - corner - y.abs()).min(0.0);
    let curved = half_w - corner + (corner * corner - delta * delta).max(0.0).sqrt();
    let factor = (0.5f32).sqrt() / sigma;
    let int_neg = 0.5 + 0.5 * erf_scalar((x - curved) * factor);
    let int_pos = 0.5 + 0.5 * erf_scalar((x + curved) * factor);
    int_pos - int_neg
}

fn rounded_box_shadow(
    lower_x: f32,
    lower_y: f32,
    upper_x: f32,
    upper_y: f32,
    pt_x: f32,
    pt_y: f32,
    sigma: f32,
    corner: f32,
) -> f32 {
    let center_x = (lower_x + upper_x) * 0.5;
    let center_y = (lower_y + upper_y) * 0.5;
    let half_w = (upper_x - lower_x) * 0.5;
    let half_h = (upper_y - lower_y) * 0.5;
    let p_x = pt_x - center_x;
    let p_y = pt_y - center_y;

    let low = p_y - half_h;
    let high = p_y + half_h;
    let start = (-3.0 * sigma).clamp(low, high);
    let end = (3.0 * sigma).clamp(low, high);

    let step = (end - start) / 4.0;
    let mut y = start + step * 0.5;
    let mut value = 0.0f32;
    for _ in 0..4 {
        value += rounded_box_shadow_x(p_x, p_y - y, sigma, corner, half_w, half_h)
            * gaussian(y, sigma)
            * step;
        y += step;
    }
    value
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rounding_alpha(coord_x: f32, coord_y: f32, size_w: f32, size_h: f32, radii: [f32; 4]) -> f32 {
    let (center_x, center_y, radius) = if coord_x < radii[0] && coord_y < radii[0] {
        (radii[0], radii[0], radii[0])
    } else if size_w - radii[1] < coord_x && coord_y < radii[1] {
        (size_w - radii[1], radii[1], radii[1])
    } else if size_w - radii[2] < coord_x && size_h - radii[2] < coord_y {
        (size_w - radii[2], size_h - radii[2], radii[2])
    } else if coord_x < radii[3] && size_h - radii[3] < coord_y {
        (radii[3], size_h - radii[3], radii[3])
    } else {
        return 1.0;
    };

    if radius <= 0.0 {
        return 1.0;
    }
    let dx = coord_x - center_x;
    let dy = coord_y - center_y;
    let dist = (dx * dx + dy * dy).sqrt();
    1.0 - smoothstep(radius - 0.5, radius + 0.5, dist)
}

/// Bakes a 9-patch template texture containing corner curvatures and 1-pixel middle slices for edges.
/// Returns (pixels, size, (margin_l, margin_r, margin_t, margin_b)).
pub fn generate_nine_patch_shadow_bitmap(
    radius: [u8; 4],
    scale: f64,
    dark_mode: bool,
) -> (Vec<u8>, Size<i32, BufferCoords>, (i32, i32, i32, i32)) {
    let sigma = (25.0 / 2.0) * scale;
    let blur_padding = (sigma * 3.0).ceil() as i32;
    let spread_phys = (5.0 * scale).round() as i32;
    let offset_y_phys = (5.0 * scale).round() as i32;

    let r_phys = radius.map(|r| (r as f64 * scale).round() as i32);
    let max_r = r_phys.iter().copied().max().unwrap_or(0);

    let margin_l = blur_padding + spread_phys + max_r;
    let margin_r = blur_padding + spread_phys + max_r;
    let margin_t = blur_padding + spread_phys - offset_y_phys + max_r;
    let margin_b = blur_padding + spread_phys + offset_y_phys + max_r;

    let tex_w = margin_l + 1 + margin_r;
    let tex_h = margin_t + 1 + margin_b;

    let base_alpha = if dark_mode { 0.45f32 } else { 0.35f32 };

    // We evaluate corners and edge slices against a large virtual box.
    // BOX_SPAN >= 800 ensures that opposite edges/corners are >= 800px away (> 60 * sigma),
    // which completely isolates the quadrants and ensures the 1-pixel edge slices represent
    // exact 1D Gaussian edge profiles with 100% full density.
    const BOX_SPAN: f32 = 800.0;

    let mut pixels = Vec::with_capacity((tex_w * tex_h * 4) as usize);

    for py in 0..tex_h {
        let pt_y = py as f32 + 0.5;

        // Determine vertical quadrant: top or bottom
        let is_bottom = py > margin_t;
        let (box_lower_y, box_upper_y, win_lower_y, win_upper_y) = if !is_bottom {
            let b_low = blur_padding as f32;
            let b_high = b_low + BOX_SPAN;
            let w_low = (margin_t - max_r) as f32;
            let w_high = w_low + BOX_SPAN;
            (b_low, b_high, w_low, w_high)
        } else {
            let b_high = (tex_h - blur_padding) as f32;
            let b_low = b_high - BOX_SPAN;
            let w_high = (tex_h - (margin_b - max_r)) as f32;
            let w_low = w_high - BOX_SPAN;
            (b_low, b_high, w_low, w_high)
        };

        for px in 0..tex_w {
            let pt_x = px as f32 + 0.5;

            // Determine horizontal quadrant: left or right
            let is_right = px > margin_l;
            let (box_lower_x, box_upper_x, win_lower_x, win_upper_x) = if !is_right {
                let b_low = blur_padding as f32;
                let b_high = b_low + BOX_SPAN;
                let w_low = (margin_l - max_r) as f32;
                let w_high = w_low + BOX_SPAN;
                (b_low, b_high, w_low, w_high)
            } else {
                let b_high = (tex_w - blur_padding) as f32;
                let b_low = b_high - BOX_SPAN;
                let w_high = (tex_w - (margin_r - max_r)) as f32;
                let w_low = w_high - BOX_SPAN;
                (b_low, b_high, w_low, w_high)
            };

            // Corner index in radius array: 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
            let corner_idx = match (is_right, is_bottom) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            };

            let corner_r = if r_phys[corner_idx] > 0 {
                (r_phys[corner_idx] + spread_phys) as f32
            } else {
                0.0
            };

            let shadow_val = if sigma < 0.1 {
                rounding_alpha(
                    pt_x - box_lower_x,
                    pt_y - box_lower_y,
                    BOX_SPAN,
                    BOX_SPAN,
                    [corner_r; 4],
                )
            } else {
                rounded_box_shadow(
                    box_lower_x,
                    box_lower_y,
                    box_upper_x,
                    box_upper_y,
                    pt_x,
                    pt_y,
                    sigma as f32,
                    corner_r,
                )
            };

            let in_win = pt_x >= win_lower_x
                && pt_x <= win_upper_x
                && pt_y >= win_lower_y
                && pt_y <= win_upper_y;

            let win_alpha = if in_win {
                let win_w = win_upper_x - win_lower_x;
                let win_h = win_upper_y - win_lower_y;
                rounding_alpha(
                    pt_x - win_lower_x,
                    pt_y - win_lower_y,
                    win_w,
                    win_h,
                    [
                        r_phys[0] as f32,
                        r_phys[1] as f32,
                        r_phys[2] as f32,
                        r_phys[3] as f32,
                    ],
                )
            } else {
                0.0
            };

            let val = shadow_val * (1.0 - win_alpha);
            let a = (val * base_alpha * 255.0).clamp(0.0, 255.0).round() as u8;

            // Abgr8888 memory representation: [R, G, B, A] = [0, 0, 0, a]
            pixels.push(0);
            pixels.push(0);
            pixels.push(0);
            pixels.push(a);
        }
    }

    (
        pixels,
        Size::from((tex_w, tex_h)),
        (margin_l, margin_r, margin_t, margin_b),
    )
}

#[derive(Debug, Clone)]
pub struct NinePatchShadowElement<T> {
    id: Id,
    commit: CommitCounter,
    texture: T,
    texture_size: Size<i32, BufferCoords>,
    area: Rectangle<i32, Logical>,
    alpha: f32,
    margin_l: i32,
    margin_r: i32,
    margin_t: i32,
    margin_b: i32,
}

impl<T: Texture + 'static> Element for NinePatchShadowElement<T> {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.commit
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size(self.area.size.to_f64().to_buffer(1.0, Transform::Normal))
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.area.to_physical_precise_round(scale)
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        self.geometry(scale).loc
    }

    fn transform(&self) -> Transform {
        Transform::Normal
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        if commit != Some(self.current_commit()) {
            DamageSet::from_slice(&[Rectangle::from_size(self.geometry(scale).size)])
        } else {
            DamageSet::default()
        }
    }

    fn opaque_regions(&self, _scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        OpaqueRegions::default()
    }

    fn alpha(&self) -> f32 {
        self.alpha
    }

    fn kind(&self) -> Kind {
        Kind::Unspecified
    }

    fn is_framebuffer_effect(&self) -> bool {
        false
    }
}

impl<R> RenderElement<R> for NinePatchShadowElement<R::TextureId>
where
    R: Renderer,
    R::TextureId: Send + 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        let dst_w = dst.size.w;
        let dst_h = dst.size.h;

        let ml = std::cmp::min(self.margin_l, dst_w / 2);
        let mr = std::cmp::min(self.margin_r, dst_w - ml);
        let mt = std::cmp::min(self.margin_t, dst_h / 2);
        let mb = std::cmp::min(self.margin_b, dst_h - mt);

        let edge_w = dst_w - ml - mr;
        let edge_h = dst_h - mt - mb;

        let x0 = dst.loc.x;
        let x1 = dst.loc.x + ml;
        let x2 = dst.loc.x + dst_w - mr;

        let y0 = dst.loc.y;
        let y1 = dst.loc.y + mt;
        let y2 = dst.loc.y + dst_h - mb;

        let u0 = 0.0;
        let u1 = self.margin_l as f64;
        let u2 = (self.margin_l + 1) as f64;
        let u3 = self.texture_size.w as f64;

        let v0 = 0.0;
        let v1 = self.margin_t as f64;
        let v2 = (self.margin_t + 1) as f64;
        let v3 = self.texture_size.h as f64;

        let render_sub_quad = |frame: &mut R::Frame<'_, '_>,
                               src_rect: Rectangle<f64, BufferCoords>,
                               quad_dst: Rectangle<i32, Physical>|
         -> Result<(), R::Error> {
            let quad_rel = Rectangle::new(quad_dst.loc - dst.loc, quad_dst.size);
            let mut local_damage = Vec::with_capacity(damage.len());
            for d in damage {
                if let Some(intersection) = d.intersection(quad_rel) {
                    local_damage.push(Rectangle::new(
                        intersection.loc - quad_rel.loc,
                        intersection.size,
                    ));
                }
            }
            if local_damage.is_empty() {
                return Ok(());
            }
            frame.render_texture_from_to(
                &self.texture,
                src_rect,
                quad_dst,
                &local_damage,
                &[],
                Transform::Normal,
                self.alpha,
            )
        };

        // 1. Top-Left Corner
        if ml > 0 && mt > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u0, v0).into(), (u1 - u0, v1 - v0).into()),
                Rectangle::new((x0, y0).into(), (ml, mt).into()),
            )?;
        }

        // 2. Top Edge
        if edge_w > 0 && mt > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u1, v0).into(), (1.0, v1 - v0).into()),
                Rectangle::new((x1, y0).into(), (edge_w, mt).into()),
            )?;
        }

        // 3. Top-Right Corner
        if mr > 0 && mt > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u2, v0).into(), (u3 - u2, v1 - v0).into()),
                Rectangle::new((x2, y0).into(), (mr, mt).into()),
            )?;
        }

        // 4. Left Edge
        if edge_h > 0 && ml > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u0, v1).into(), (u1 - u0, 1.0).into()),
                Rectangle::new((x0, y1).into(), (ml, edge_h).into()),
            )?;
        }

        // 5. Right Edge
        if edge_h > 0 && mr > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u2, v1).into(), (u3 - u2, 1.0).into()),
                Rectangle::new((x2, y1).into(), (mr, edge_h).into()),
            )?;
        }

        // 6. Bottom-Left Corner
        if ml > 0 && mb > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u0, v2).into(), (u1 - u0, v3 - v2).into()),
                Rectangle::new((x0, y2).into(), (ml, mb).into()),
            )?;
        }

        // 7. Bottom Edge
        if edge_w > 0 && mb > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u1, v2).into(), (1.0, v3 - v2).into()),
                Rectangle::new((x1, y2).into(), (edge_w, mb).into()),
            )?;
        }

        // 8. Bottom-Right Corner
        if mr > 0 && mb > 0 {
            render_sub_quad(
                frame,
                Rectangle::new((u2, v2).into(), (u3 - u2, v3 - v2).into()),
                Rectangle::new((x2, y2).into(), (mr, mb).into()),
            )?;
        }

        Ok(())
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }

    fn capture_framebuffer(
        &self,
        _frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        _dst: Rectangle<i32, Physical>,
        _cache: &UserDataMap,
    ) -> Result<(), R::Error> {
        Ok(())
    }

    fn prepare_texture(&self, frame: &mut R::Frame<'_, '_>) -> Result<(), R::Error> {
        frame.prepare_textures(std::iter::once(&self.texture))
    }
}

#[derive(Debug, Clone)]
pub enum ShadowElement<R: Renderer> {
    NinePatch(NinePatchShadowElement<R::TextureId>),
    Pixel(PixelShaderElement),
}

impl<R> Element for ShadowElement<R>
where
    R: Renderer,
    R::TextureId: 'static,
{
    fn id(&self) -> &Id {
        match self {
            ShadowElement::NinePatch(elem) => elem.id(),
            ShadowElement::Pixel(elem) => elem.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            ShadowElement::NinePatch(elem) => elem.current_commit(),
            ShadowElement::Pixel(elem) => elem.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        match self {
            ShadowElement::NinePatch(elem) => elem.src(),
            ShadowElement::Pixel(elem) => elem.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            ShadowElement::NinePatch(elem) => elem.geometry(scale),
            ShadowElement::Pixel(elem) => elem.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            ShadowElement::NinePatch(elem) => elem.location(scale),
            ShadowElement::Pixel(elem) => elem.location(scale),
        }
    }

    fn transform(&self) -> Transform {
        match self {
            ShadowElement::NinePatch(elem) => elem.transform(),
            ShadowElement::Pixel(elem) => elem.transform(),
        }
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        match self {
            ShadowElement::NinePatch(elem) => elem.damage_since(scale, commit),
            ShadowElement::Pixel(elem) => elem.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            ShadowElement::NinePatch(elem) => elem.opaque_regions(scale),
            ShadowElement::Pixel(elem) => elem.opaque_regions(scale),
        }
    }

    fn alpha(&self) -> f32 {
        match self {
            ShadowElement::NinePatch(elem) => elem.alpha(),
            ShadowElement::Pixel(elem) => elem.alpha(),
        }
    }

    fn kind(&self) -> Kind {
        match self {
            ShadowElement::NinePatch(elem) => elem.kind(),
            ShadowElement::Pixel(elem) => elem.kind(),
        }
    }

    fn is_framebuffer_effect(&self) -> bool {
        match self {
            ShadowElement::NinePatch(elem) => elem.is_framebuffer_effect(),
            ShadowElement::Pixel(elem) => elem.is_framebuffer_effect(),
        }
    }
}

impl<R> RenderElement<R> for ShadowElement<R>
where
    R: Renderer + ImportAll + ImportMem + AsGlowRenderer,
    R::TextureId: Send + 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        match self {
            ShadowElement::NinePatch(elem) => {
                RenderElement::<R>::draw(elem, frame, src, dst, damage, opaque_regions, cache)
            }
            ShadowElement::Pixel(elem) => {
                if let Some(glow_frame) = R::glow_frame_mut(frame) {
                    RenderElement::<GlowRenderer>::draw(
                        elem,
                        glow_frame,
                        src,
                        dst,
                        damage,
                        opaque_regions,
                        cache,
                    )
                    .map_err(R::from_gles_error)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn underlying_storage(&self, renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        match self {
            ShadowElement::NinePatch(elem) => {
                RenderElement::<R>::underlying_storage(elem, renderer)
            }
            ShadowElement::Pixel(elem) => renderer
                .glow_renderer_mut()
                .and_then(|glow| elem.underlying_storage(glow)),
        }
    }

    fn capture_framebuffer(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), R::Error> {
        match self {
            ShadowElement::NinePatch(elem) => {
                RenderElement::<R>::capture_framebuffer(elem, frame, src, dst, cache)
            }
            ShadowElement::Pixel(elem) => {
                if let Some(glow_frame) = R::glow_frame_mut(frame) {
                    RenderElement::<GlowRenderer>::capture_framebuffer(
                        elem, glow_frame, src, dst, cache,
                    )
                    .map_err(R::from_gles_error)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn prepare_texture(&self, frame: &mut R::Frame<'_, '_>) -> Result<(), R::Error> {
        match self {
            ShadowElement::NinePatch(elem) => RenderElement::<R>::prepare_texture(elem, frame),
            ShadowElement::Pixel(elem) => {
                if let Some(glow_frame) = R::glow_frame_mut(frame) {
                    RenderElement::<GlowRenderer>::prepare_texture(elem, glow_frame)
                        .map_err(R::from_gles_error)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ShadowShader {
    pub fn get<R: AsGlowRenderer>(renderer: &R) -> GlesPixelProgram {
        let Some(glow) = renderer.glow_renderer() else {
            return GlesPixelProgram::dummy();
        };
        Borrow::<GlesRenderer>::borrow(glow)
            .egl_context()
            .user_data()
            .get::<ShadowShader>()
            .expect("Custom Shaders not initialized")
            .0
            .clone()
    }

    pub fn pixel_shader_element<R: AsGlowRenderer>(
        renderer: &R,
        key: CosmicMappedKey,
        geo: Rectangle<i32, Local>,
        radius: [u8; 4],
        alpha: f32,
        scale: f64,
        dark_mode: bool,
    ) -> PixelShaderElement {
        let Some(glow) = renderer.glow_renderer() else {
            return PixelShaderElement::dummy();
        };

        let params = ShadowParameters {
            geo,
            scale,
            alpha,
            radius,
            dark_mode,
        };
        let ceil = |logical: f64| (logical * scale).ceil() / scale;

        let mut geo = geo.to_f64();
        let fractional_pixel = scale.ceil() / scale;
        geo.loc.x += fractional_pixel;
        geo.loc.y += fractional_pixel;
        geo.size.w -= fractional_pixel * 2.;
        geo.size.h -= fractional_pixel * 2.;

        let user_data = Borrow::<GlesRenderer>::borrow(glow)
            .egl_context()
            .user_data();

        user_data.insert_if_missing(|| ShadowCache::new(HashMap::new()));
        let mut cache = user_data.get::<ShadowCache>().unwrap().borrow_mut();
        cache.retain(|k, _| k.alive());

        if cache
            .get(&key)
            .filter(|(old_params, _)| &params == old_params)
            .is_none()
        {
            let shader = Self::get(renderer);

            let softness: f64 = 25.;
            let spread: f64 = 5.;
            let offset = [0., 5.];
            let color = [0., 0., 0., if dark_mode { 0.45 } else { 0.35 }];
            let radius = radius.map(|r| ceil(r as f64));

            let width = softness;
            let sigma = width / 2.;
            let width = ceil(sigma * 3.);

            let offset = Point::new(ceil(offset[0]), ceil(offset[1]));
            let spread = ceil(spread.abs()).copysign(spread);
            let offset = offset - Point::new(spread, spread);

            let box_size = if spread >= 0. {
                geo.size + Size::new(spread, spread).upscale(2.)
            } else {
                geo.size - Size::new(-spread, -spread).upscale(2.)
            };

            let win_radius = radius;
            let radius = radius.map(|r| {
                if r > 0. {
                    smithay::utils::Coordinate::saturating_add(r, spread)
                } else {
                    0.
                }
            });
            let shader_size = box_size + Size::from((width, width)).upscale(2.);
            let mut shader_geo = Rectangle::new(Point::from((-width, -width)), shader_size);

            let window_geo = Rectangle::new(Point::new(0., 0.) - offset - shader_geo.loc, geo.size);
            let area_size = Vec2::new(shader_geo.size.w as f32, shader_geo.size.h as f32);
            let geo_loc = Vec2::new(-shader_geo.loc.x as f32, -shader_geo.loc.y as f32);
            shader_geo.loc += offset + geo.loc;

            let input_to_geo = Mat3::from(
                Affine2::from_scale(area_size)
                    * Affine2::from_translation(Vec2::new(
                        -geo_loc.x / area_size.x,
                        -geo_loc.y / area_size.y,
                    )),
            );

            let window_geo_loc = Vec2::new(window_geo.loc.x as f32, window_geo.loc.y as f32);
            let window_input_to_geo = Mat3::from(
                Affine2::from_scale(area_size)
                    * Affine2::from_translation(Vec2::new(
                        -window_geo_loc.x / area_size.x,
                        -window_geo_loc.y / area_size.y,
                    )),
            );

            let element = PixelShaderElement::new(
                shader,
                shader_geo.to_i32_up().as_logical(),
                None,
                alpha,
                vec![
                    Uniform::new("shadow_color", color),
                    Uniform::new("sigma", sigma as f32),
                    Uniform::new(
                        "input_to_geo",
                        UniformValue::Matrix3x3 {
                            matrices: vec![*AsRef::<[f32; 9]>::as_ref(&input_to_geo)],
                            transpose: false,
                        },
                    ),
                    Uniform::new("geo_size", [box_size.w as f32, box_size.h as f32]),
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
                        "window_input_to_geo",
                        UniformValue::Matrix3x3 {
                            matrices: vec![*AsRef::<[f32; 9]>::as_ref(&window_input_to_geo)],
                            transpose: false,
                        },
                    ),
                    Uniform::new(
                        "window_geo_size",
                        [window_geo.size.w as f32, window_geo.size.h as f32],
                    ),
                    Uniform::new(
                        "window_corner_radius",
                        [
                            win_radius[0] as f32,
                            win_radius[1] as f32,
                            win_radius[2] as f32,
                            win_radius[3] as f32,
                        ],
                    ),
                ],
                Kind::Unspecified,
            );

            cache.insert(key.clone(), (params, element));
        }

        cache.get(&key).unwrap().1.clone()
    }

    pub fn nine_patch_element<R>(
        renderer: &mut R,
        key: CosmicMappedKey,
        geo: Rectangle<i32, Local>,
        radius: [u8; 4],
        alpha: f32,
        scale: f64,
        dark_mode: bool,
    ) -> Result<NinePatchShadowElement<R::TextureId>, R::Error>
    where
        R: Renderer + ImportMem,
        R::TextureId: Clone + Send + 'static,
    {
        let params = ShadowParameters {
            geo,
            scale,
            alpha,
            radius,
            dark_mode,
        };

        let context_id = renderer.context_id().erased();
        let tex_key = NinePatchKey {
            context_id,
            radius,
            scale_bits: scale.to_bits(),
            dark_mode,
        };

        let (texture, tex_size, margins) = {
            let cached = NINE_PATCH_CACHE.with(|cache| {
                cache.borrow().get(&tex_key).and_then(|c| {
                    c.texture
                        .downcast_ref::<R::TextureId>()
                        .map(|t| (t.clone(), c.size, c.margins))
                })
            });

            if let Some(res) = cached {
                res
            } else {
                let (pixels, sz, margins) =
                    generate_nine_patch_shadow_bitmap(radius, scale, dark_mode);
                let tex = renderer.import_memory(&pixels, Fourcc::Abgr8888, sz, false)?;
                NINE_PATCH_CACHE.with(|cache| {
                    cache.borrow_mut().insert(
                        tex_key,
                        CachedNinePatchTexture {
                            texture: Box::new(tex.clone()),
                            size: sz,
                            margins,
                        },
                    );
                });
                (tex, sz, margins)
            }
        };

        let ceil = |logical: f64| (logical * scale).ceil() / scale;

        let geo_f64 = geo.to_f64();

        let softness: f64 = 25.;
        let spread: f64 = 5.;
        let offset = [0., 5.];

        let width = softness;
        let sigma = width / 2.;
        let width = ceil(sigma * 3.);

        let offset = Point::new(ceil(offset[0]), ceil(offset[1]));
        let spread = ceil(spread.abs()).copysign(spread);
        let offset = offset - Point::new(spread, spread);

        let box_size = if spread >= 0. {
            geo_f64.size + Size::new(spread, spread).upscale(2.)
        } else {
            geo_f64.size - Size::new(-spread, -spread).upscale(2.)
        };

        let shader_size = box_size + Size::from((width, width)).upscale(2.);
        let mut shader_geo = Rectangle::new(Point::from((-width, -width)), shader_size);
        shader_geo.loc += offset + geo_f64.loc;
        let area = shader_geo.to_i32_up().as_logical();

        let (id, commit) = NINE_PATCH_ELEMENT_CACHE.with(|elem_cache| {
            let mut cache = elem_cache.borrow_mut();
            cache.retain(|k, _| k.alive());

            if let Some(entry) = cache.get_mut(&key) {
                if entry.params != params {
                    entry.params = params;
                    entry.commit.increment();
                }
                (entry.id.clone(), entry.commit)
            } else {
                let id = Id::new();
                let commit = CommitCounter::default();
                cache.insert(
                    key.clone(),
                    CachedNinePatchWindow {
                        params,
                        id: id.clone(),
                        commit,
                    },
                );
                (id, commit)
            }
        });

        Ok(NinePatchShadowElement {
            id,
            commit,
            texture,
            texture_size: tex_size,
            area,
            alpha,
            margin_l: margins.0,
            margin_r: margins.1,
            margin_t: margins.2,
            margin_b: margins.3,
        })
    }

    pub fn element<R>(
        renderer: &mut R,
        key: CosmicMappedKey,
        geo: Rectangle<i32, Local>,
        radius: [u8; 4],
        alpha: f32,
        scale: f64,
        dark_mode: bool,
    ) -> ShadowElement<R>
    where
        R: Renderer + ImportAll + ImportMem + AsGlowRenderer,
        R::TextureId: Clone + Send + 'static,
    {
        let prefer_pixel = renderer.glow_renderer().is_some() && !*FORCE_NINEPATCH;

        if !prefer_pixel {
            match Self::nine_patch_element(
                renderer,
                key.clone(),
                geo,
                radius,
                alpha,
                scale,
                dark_mode,
            ) {
                Ok(elem) => return ShadowElement::NinePatch(elem),
                Err(err) => {
                    tracing::warn!("Failed to create 9-patch shadow element: {:?}", err);
                }
            }
        }

        ShadowElement::Pixel(Self::pixel_shader_element(
            renderer, key, geo, radius, alpha, scale, dark_mode,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nine_patch_shadow_bitmap_generation() {
        let radius = [8, 8, 8, 8];
        let scale = 1.0;
        let dark_mode = true;
        let (pixels, size, margins) = generate_nine_patch_shadow_bitmap(radius, scale, dark_mode);

        assert_eq!(size.w, margins.0 + 1 + margins.1);
        assert_eq!(size.h, margins.2 + 1 + margins.3);
        assert_eq!(pixels.len(), (size.w * size.h * 4) as usize);

        // Center column and row (inside window) should be cut out (alpha = 0)
        let cx = margins.0;
        let cy = margins.2;
        let idx = ((cy * size.w + cx) * 4) as usize;
        assert_eq!(
            pixels[idx + 3],
            0,
            "Center pixel should be cut out (alpha = 0)"
        );

        // Outer blur region should have non-zero alpha
        let outer_idx = ((10 * size.w + cx) * 4) as usize;
        assert!(
            pixels[outer_idx + 3] > 0,
            "Outer blur pixel should have non-zero alpha"
        );
    }

    #[test]
    fn test_nine_patch_ssd_corners() {
        // Window with SSD: top corners rounded, bottom corners square
        let radius = [12, 12, 0, 0];
        let scale = 1.0;
        let dark_mode = false;
        let (pixels, size, margins) = generate_nine_patch_shadow_bitmap(radius, scale, dark_mode);

        assert_eq!(size.w, margins.0 + 1 + margins.1);
        assert_eq!(size.h, margins.2 + 1 + margins.3);
        assert_eq!(pixels.len(), (size.w * size.h * 4) as usize);

        // Top corner margin should include max_r (12)
        assert_eq!(margins.2, 38 + 12);
        // Bottom corner margin should also include max_r (12) for layout symmetry
        assert_eq!(margins.3, 38 + 10 + 12);
    }

    #[test]
    fn test_nine_patch_hidpi_scaling() {
        // HiDPI scale 2.0
        let radius = [8, 8, 8, 8];
        let scale = 2.0;
        let dark_mode = true;
        let (pixels, size, margins) = generate_nine_patch_shadow_bitmap(radius, scale, dark_mode);

        assert_eq!(size.w, margins.0 + 1 + margins.1);
        assert_eq!(size.h, margins.2 + 1 + margins.3);
        assert_eq!(pixels.len(), (size.w * size.h * 4) as usize);

        // At scale 2.0:
        // sigma = 12.5 * 2 = 25.0, blur_padding = ceil(25 * 3) = 75
        // spread_phys = 10, offset_y_phys = 10, max_r = 16
        // margin_l = 75 + 10 + 16 = 101
        // margin_r = 101
        // margin_t = 75 + 16 = 91
        // margin_b = 75 + 20 + 16 = 111
        assert_eq!(margins.0, 101);
        assert_eq!(margins.1, 101);
        assert_eq!(margins.2, 91);
        assert_eq!(margins.3, 111);
        assert_eq!(size.w, 101 + 1 + 101);
        assert_eq!(size.h, 91 + 1 + 111);
    }

    #[test]
    fn test_nine_patch_quad_coverage() {
        // Ensure that 8 quads + center hole completely and without gaps tile the destination geometry
        let dst_x = 100;
        let dst_y = 150;
        let dst_w = 800;
        let dst_h = 600;
        let margin_l = 51;
        let margin_r = 51;
        let margin_t = 46;
        let margin_b = 56;

        let ml = std::cmp::min(margin_l, dst_w / 2);
        let mr = std::cmp::min(margin_r, dst_w - ml);
        let mt = std::cmp::min(margin_t, dst_h / 2);
        let mb = std::cmp::min(margin_b, dst_h - mt);

        let edge_w = dst_w - ml - mr;
        let edge_h = dst_h - mt - mb;

        // Check horizontal continuity
        assert_eq!(ml + edge_w + mr, dst_w);
        // Check vertical continuity
        assert_eq!(mt + edge_h + mb, dst_h);

        let x0 = dst_x;
        let x1 = dst_x + ml;
        let x2 = dst_x + dst_w - mr;
        let x3 = dst_x + dst_w;

        let y0 = dst_y;
        let y1 = dst_y + mt;
        let y2 = dst_y + dst_h - mb;
        let _y3 = dst_y + dst_h;

        // Top-left quad
        assert_eq!(x1 - x0, ml);
        assert_eq!(y1 - y0, mt);
        // Top edge quad
        assert_eq!(x2 - x1, edge_w);
        assert_eq!(y1 - y0, mt);
        // Top-right quad
        assert_eq!(x3 - x2, mr);
        assert_eq!(y1 - y0, mt);

        // Center hole (unrendered window interior)
        assert_eq!(x2 - x1, edge_w);
        assert_eq!(y2 - y1, edge_h);
    }

    #[test]
    fn test_nine_patch_square_corners() {
        let radius = [0, 0, 0, 0];
        let scale = 1.0;
        let dark_mode = true;
        let (pixels, size, margins) = generate_nine_patch_shadow_bitmap(radius, scale, dark_mode);

        assert_eq!(margins.0, 43);
        assert_eq!(margins.1, 43);
        assert_eq!(margins.2, 38);
        assert_eq!(margins.3, 48);
        assert_eq!(size.w, 43 + 1 + 43);
        assert_eq!(size.h, 38 + 1 + 48);
        assert_eq!(pixels.len(), (87 * 87 * 4) as usize);

        let cx = margins.0;
        let cy = margins.2;
        let idx = ((cy * size.w + cx) * 4) as usize;
        assert_eq!(pixels[idx + 3], 0, "Center pixel should be cut out");
    }

    #[test]
    fn test_nine_patch_damage_culling_and_translation() {
        let dst = Rectangle::new(Point::from((100, 100)), Size::from((800, 600)));
        let ml = 51;
        let mr = 51;
        let mt = 46;
        let mb = 56;
        let edge_w = 800 - ml - mr; // 698
        let _edge_h = 600 - mt - mb; // 498

        // Define top-left and top-edge quad destinations
        let quad_tl_dst = Rectangle::new(dst.loc, Size::from((ml, mt)));
        let quad_top_dst = Rectangle::new(
            Point::from((dst.loc.x + ml, dst.loc.y)),
            Size::from((edge_w, mt)),
        );

        // Helper replicating render_sub_quad damage calculation
        let compute_local_damage =
            |quad_dst: Rectangle<i32, Physical>, damage: &[Rectangle<i32, Physical>]| {
                let quad_rel = Rectangle::new(quad_dst.loc - dst.loc, quad_dst.size);
                let mut local_damage = Vec::new();
                for d in damage {
                    if let Some(intersection) = d.intersection(quad_rel) {
                        local_damage.push(Rectangle::new(
                            intersection.loc - quad_rel.loc,
                            intersection.size,
                        ));
                    }
                }
                local_damage
            };

        // Scenario 1: Damage in top-left corner (0, 0, 30, 30) relative to dst.loc
        let damage1 = [Rectangle::new(Point::from((0, 0)), Size::from((30, 30)))];
        let tl_dmg1 = compute_local_damage(quad_tl_dst, &damage1);
        let top_dmg1 = compute_local_damage(quad_top_dst, &damage1);

        assert_eq!(tl_dmg1.len(), 1);
        assert_eq!(
            tl_dmg1[0],
            Rectangle::new(Point::from((0, 0)), Size::from((30, 30)))
        );
        assert!(
            top_dmg1.is_empty(),
            "Top edge quad should cull damage outside its bounds"
        );

        // Scenario 2: Damage on top edge at (200, 10, 40, 20) relative to dst.loc
        let damage2 = [Rectangle::new(Point::from((200, 10)), Size::from((40, 20)))];
        let tl_dmg2 = compute_local_damage(quad_tl_dst, &damage2);
        let top_dmg2 = compute_local_damage(quad_top_dst, &damage2);

        assert!(
            tl_dmg2.is_empty(),
            "Top-left quad should cull damage on top edge"
        );
        assert_eq!(top_dmg2.len(), 1);
        // Translation: 200 - ml (51) = 149
        assert_eq!(
            top_dmg2[0],
            Rectangle::new(Point::from((149, 10)), Size::from((40, 20)))
        );
    }
}
