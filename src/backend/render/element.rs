use crate::{
    backend::{
        kms::render::gles::GbmGlowBackend,
        render::{GlMultiError, VulkanMultiError, wayland::SurfaceRenderElement},
    },
    shell::{CosmicMappedRenderElement, WorkspaceRenderElement},
    utils::iced::IcedRenderElement,
};

#[cfg(feature = "debug")]
use smithay::backend::renderer::element::texture::TextureRenderElement;
use smithay::{
    backend::{
        allocator::{Fourcc, dmabuf::Dmabuf},
        drm::DrmDeviceFd,
        renderer::{
            Bind, Blit, ContextId, ExportMem, ImportAll, ImportMem, Offscreen, Renderer,
            TextureFilter,
            element::{
                Element, Id, Kind, RenderElement, UnderlyingStorage,
                utils::{CropRenderElement, Relocate, RelocateRenderElement, RescaleRenderElement},
            },
            gles::{GlesError, GlesRenderbuffer, GlesTexture, element::TextureShaderElement},
            glow::{GlowFrame, GlowRenderer},
            multigpu::MultiTexture,
            sync::SyncPoint,
            utils::{CommitCounter, DamageSet, OpaqueRegions},
        },
    },
    utils::{
        Buffer as BufferCoords, Logical, Physical, Point, Rectangle, Scale, Size,
        user_data::UserDataMap,
    },
};

use super::{GlMultiRenderer, cursor::CursorRenderElement};

pub enum CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    Workspace(
        RelocateRenderElement<CropRenderElement<RescaleRenderElement<WorkspaceRenderElement<R>>>>,
    ),
    Cursor(
        RescaleRenderElement<RescaleRenderElement<RelocateRenderElement<CursorRenderElement<R>>>>,
    ),
    Dnd(SurfaceRenderElement<R>),
    MoveGrab(RescaleRenderElement<CosmicMappedRenderElement<R>>),
    Postprocess(
        CropRenderElement<RelocateRenderElement<RescaleRenderElement<TextureShaderElement>>>,
    ),
    Zoom(IcedRenderElement<R>),
    Damage(DamageElement),
    #[cfg(feature = "debug")]
    Egui(TextureRenderElement<GlesTexture>),
}

impl<R> Element for CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    fn id(&self) -> &Id {
        match self {
            CosmicElement::Workspace(elem) => elem.id(),
            CosmicElement::Cursor(elem) => elem.id(),
            CosmicElement::Dnd(elem) => elem.id(),
            CosmicElement::MoveGrab(elem) => elem.id(),
            CosmicElement::Postprocess(elem) => elem.id(),
            CosmicElement::Zoom(elem) => elem.id(),
            CosmicElement::Damage(elem) => elem.id(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            CosmicElement::Workspace(elem) => elem.current_commit(),
            CosmicElement::Cursor(elem) => elem.current_commit(),
            CosmicElement::Dnd(elem) => elem.current_commit(),
            CosmicElement::MoveGrab(elem) => elem.current_commit(),
            CosmicElement::Postprocess(elem) => elem.current_commit(),
            CosmicElement::Zoom(elem) => elem.current_commit(),
            CosmicElement::Damage(elem) => elem.current_commit(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, smithay::utils::Buffer> {
        match self {
            CosmicElement::Workspace(elem) => elem.src(),
            CosmicElement::Cursor(elem) => elem.src(),
            CosmicElement::Dnd(elem) => elem.src(),
            CosmicElement::MoveGrab(elem) => elem.src(),
            CosmicElement::Postprocess(elem) => elem.src(),
            CosmicElement::Zoom(elem) => elem.src(),
            CosmicElement::Damage(elem) => elem.src(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            CosmicElement::Workspace(elem) => elem.geometry(scale),
            CosmicElement::Cursor(elem) => elem.geometry(scale),
            CosmicElement::Dnd(elem) => elem.geometry(scale),
            CosmicElement::MoveGrab(elem) => elem.geometry(scale),
            CosmicElement::Postprocess(elem) => elem.geometry(scale),
            CosmicElement::Zoom(elem) => elem.geometry(scale),
            CosmicElement::Damage(elem) => elem.geometry(scale),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            CosmicElement::Workspace(elem) => elem.location(scale),
            CosmicElement::Cursor(elem) => elem.location(scale),
            CosmicElement::Dnd(elem) => elem.location(scale),
            CosmicElement::MoveGrab(elem) => elem.location(scale),
            CosmicElement::Postprocess(elem) => elem.location(scale),
            CosmicElement::Zoom(elem) => elem.location(scale),
            CosmicElement::Damage(elem) => elem.location(scale),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.location(scale),
        }
    }

    fn transform(&self) -> smithay::utils::Transform {
        match self {
            CosmicElement::Workspace(elem) => elem.transform(),
            CosmicElement::Cursor(elem) => elem.transform(),
            CosmicElement::Dnd(elem) => elem.transform(),
            CosmicElement::MoveGrab(elem) => elem.transform(),
            CosmicElement::Postprocess(elem) => elem.transform(),
            CosmicElement::Zoom(elem) => elem.transform(),
            CosmicElement::Damage(elem) => elem.transform(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.transform(),
        }
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        match self {
            CosmicElement::Workspace(elem) => elem.damage_since(scale, commit),
            CosmicElement::Cursor(elem) => elem.damage_since(scale, commit),
            CosmicElement::Dnd(elem) => elem.damage_since(scale, commit),
            CosmicElement::MoveGrab(elem) => elem.damage_since(scale, commit),
            CosmicElement::Postprocess(elem) => elem.damage_since(scale, commit),
            CosmicElement::Zoom(elem) => elem.damage_since(scale, commit),
            CosmicElement::Damage(elem) => elem.damage_since(scale, commit),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            CosmicElement::Workspace(elem) => elem.opaque_regions(scale),
            CosmicElement::Cursor(elem) => elem.opaque_regions(scale),
            CosmicElement::Dnd(elem) => elem.opaque_regions(scale),
            CosmicElement::MoveGrab(elem) => elem.opaque_regions(scale),
            CosmicElement::Postprocess(elem) => elem.opaque_regions(scale),
            CosmicElement::Zoom(elem) => elem.opaque_regions(scale),
            CosmicElement::Damage(elem) => elem.opaque_regions(scale),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.opaque_regions(scale),
        }
    }

    fn alpha(&self) -> f32 {
        match self {
            CosmicElement::Workspace(elem) => elem.alpha(),
            CosmicElement::Cursor(elem) => elem.alpha(),
            CosmicElement::Dnd(elem) => elem.alpha(),
            CosmicElement::MoveGrab(elem) => elem.alpha(),
            CosmicElement::Postprocess(elem) => elem.alpha(),
            CosmicElement::Zoom(elem) => elem.alpha(),
            CosmicElement::Damage(elem) => elem.alpha(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.alpha(),
        }
    }

    fn kind(&self) -> Kind {
        match self {
            CosmicElement::Workspace(elem) => elem.kind(),
            CosmicElement::Cursor(elem) => elem.kind(),
            CosmicElement::Dnd(elem) => elem.kind(),
            CosmicElement::MoveGrab(elem) => elem.kind(),
            CosmicElement::Postprocess(elem) => elem.kind(),
            CosmicElement::Zoom(elem) => elem.kind(),
            CosmicElement::Damage(elem) => elem.kind(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.kind(),
        }
    }

    fn is_framebuffer_effect(&self) -> bool {
        match self {
            CosmicElement::Workspace(elem) => elem.is_framebuffer_effect(),
            CosmicElement::Cursor(elem) => elem.is_framebuffer_effect(),
            CosmicElement::Dnd(elem) => elem.is_framebuffer_effect(),
            CosmicElement::MoveGrab(elem) => elem.is_framebuffer_effect(),
            CosmicElement::Postprocess(elem) => elem.is_framebuffer_effect(),
            CosmicElement::Zoom(elem) => elem.is_framebuffer_effect(),
            CosmicElement::Damage(elem) => elem.is_framebuffer_effect(),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => elem.is_framebuffer_effect(),
        }
    }
}

impl<R> RenderElement<R> for CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
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
            CosmicElement::Workspace(elem) => {
                elem.draw(frame, src, dst, damage, opaque_regions, cache)
            }
            CosmicElement::Cursor(elem) => {
                elem.draw(frame, src, dst, damage, opaque_regions, cache)
            }
            CosmicElement::Dnd(elem) => elem.draw(frame, src, dst, damage, opaque_regions, cache),
            CosmicElement::MoveGrab(elem) => {
                elem.draw(frame, src, dst, damage, opaque_regions, cache)
            }
            CosmicElement::Postprocess(elem) => {
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
            CosmicElement::Zoom(elem) => elem.draw(frame, src, dst, damage, opaque_regions, cache),
            CosmicElement::Damage(elem) => {
                RenderElement::<R>::draw(elem, frame, src, dst, damage, opaque_regions, cache)
            }
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => {
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
            CosmicElement::Workspace(elem) => elem.underlying_storage(renderer),
            CosmicElement::Cursor(elem) => elem.underlying_storage(renderer),
            CosmicElement::Dnd(elem) => elem.underlying_storage(renderer),
            CosmicElement::MoveGrab(elem) => elem.underlying_storage(renderer),
            CosmicElement::Postprocess(elem) => renderer
                .glow_renderer_mut()
                .and_then(|glow_renderer| elem.underlying_storage(glow_renderer)),
            CosmicElement::Zoom(elem) => elem.underlying_storage(renderer),
            CosmicElement::Damage(elem) => elem.underlying_storage(renderer),
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => renderer
                .glow_renderer_mut()
                .and_then(|glow_renderer| elem.underlying_storage(glow_renderer)),
        }
    }

    fn capture_framebuffer(
        &self,
        frame: &mut <R>::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), <R>::Error> {
        match self {
            CosmicElement::Workspace(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CosmicElement::Cursor(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CosmicElement::Dnd(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CosmicElement::MoveGrab(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CosmicElement::Postprocess(elem) => {
                if let Some(glow_frame) = R::glow_frame_mut(frame) {
                    RenderElement::<GlowRenderer>::capture_framebuffer(
                        elem, glow_frame, src, dst, cache,
                    )
                    .map_err(R::from_gles_error)
                } else {
                    Ok(())
                }
            }
            CosmicElement::Zoom(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CosmicElement::Damage(elem) => {
                RenderElement::<R>::capture_framebuffer(elem, frame, src, dst, cache)
            }
            #[cfg(feature = "debug")]
            CosmicElement::Egui(elem) => {
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
}

impl<R> From<CropRenderElement<RescaleRenderElement<WorkspaceRenderElement<R>>>>
    for CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    fn from(elem: CropRenderElement<RescaleRenderElement<WorkspaceRenderElement<R>>>) -> Self {
        Self::Workspace(RelocateRenderElement::from_element(
            elem,
            (0, 0),
            Relocate::Relative,
        ))
    }
}

impl<R> From<IcedRenderElement<R>> for CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    fn from(value: IcedRenderElement<R>) -> Self {
        Self::Zoom(value)
    }
}

impl<R> From<DamageElement> for CosmicElement<R>
where
    R: Renderer + ImportAll + ImportMem + AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    fn from(value: DamageElement) -> Self {
        Self::Damage(value)
    }
}

#[cfg(feature = "debug")]
impl<R> From<TextureRenderElement<GlesTexture>> for CosmicElement<R>
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
    CosmicMappedRenderElement<R>: RenderElement<R>,
{
    fn from(elem: TextureRenderElement<GlesTexture>) -> Self {
        Self::Egui(elem)
    }
}

pub trait AsGlowRenderer: Renderer + ImportAll + ImportMem + ExportMem + Bind<Dmabuf> {
    fn glow_renderer(&self) -> Option<&GlowRenderer>;
    fn glow_renderer_mut(&mut self) -> Option<&mut GlowRenderer>;
    fn glow_frame<'a, 'frame, 'buffer>(
        frame: &'a Self::Frame<'frame, 'buffer>,
    ) -> Option<&'a GlowFrame<'frame, 'buffer>>;
    fn glow_frame_mut<'a, 'frame, 'buffer>(
        frame: &'a mut Self::Frame<'frame, 'buffer>,
    ) -> Option<&'a mut GlowFrame<'frame, 'buffer>>;
    fn tex_from_gl(
        context: &ContextId<GlesTexture>,
        texture: GlesTexture,
    ) -> Option<Self::TextureId>;
    fn tex_to_gl(
        context: &ContextId<GlesTexture>,
        texture: &Self::TextureId,
    ) -> Option<GlesTexture>;
    fn from_gles_error(err: GlesError) -> Self::Error;

    fn bind_glow_texture<'a>(
        &mut self,
        target: &'a mut GlesTexture,
    ) -> Result<Self::Framebuffer<'a>, Self::Error>;
    fn bind_glow_renderbuffer<'a>(
        &mut self,
        target: &'a mut GlesRenderbuffer,
    ) -> Result<Self::Framebuffer<'a>, Self::Error>;
    fn create_glow_renderbuffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoords>,
    ) -> Result<GlesRenderbuffer, Self::Error>;
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error>;
}

impl AsGlowRenderer for GlowRenderer {
    fn glow_renderer(&self) -> Option<&GlowRenderer> {
        Some(self)
    }
    fn glow_renderer_mut(&mut self) -> Option<&mut GlowRenderer> {
        Some(self)
    }
    fn glow_frame<'a, 'frame, 'buffer>(
        frame: &'a Self::Frame<'frame, 'buffer>,
    ) -> Option<&'a GlowFrame<'frame, 'buffer>> {
        Some(frame)
    }
    fn glow_frame_mut<'a, 'frame, 'buffer>(
        frame: &'a mut Self::Frame<'frame, 'buffer>,
    ) -> Option<&'a mut GlowFrame<'frame, 'buffer>> {
        Some(frame)
    }
    fn tex_from_gl(
        _context: &ContextId<GlesTexture>,
        texture: GlesTexture,
    ) -> Option<Self::TextureId> {
        Some(texture)
    }
    fn tex_to_gl(
        _context: &ContextId<GlesTexture>,
        texture: &Self::TextureId,
    ) -> Option<GlesTexture> {
        Some(texture.clone())
    }
    fn from_gles_error(err: GlesError) -> Self::Error {
        err
    }

    fn bind_glow_texture<'a>(
        &mut self,
        target: &'a mut GlesTexture,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind(target)
    }
    fn bind_glow_renderbuffer<'a>(
        &mut self,
        target: &'a mut GlesRenderbuffer,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind(target)
    }
    fn create_glow_renderbuffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoords>,
    ) -> Result<GlesRenderbuffer, Self::Error> {
        Offscreen::<GlesRenderbuffer>::create_buffer(self, format, size)
    }
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        Blit::blit(self, from, to, src, dst, filter)
    }
}

impl AsGlowRenderer for GlMultiRenderer<'_> {
    fn glow_renderer(&self) -> Option<&GlowRenderer> {
        Some(self.as_ref())
    }
    fn glow_renderer_mut(&mut self) -> Option<&mut GlowRenderer> {
        Some(self.as_mut())
    }
    fn glow_frame<'b, 'frame, 'buffer>(
        frame: &'b Self::Frame<'frame, 'buffer>,
    ) -> Option<&'b GlowFrame<'frame, 'buffer>> {
        Some(frame.as_ref())
    }
    fn glow_frame_mut<'b, 'frame, 'buffer>(
        frame: &'b mut Self::Frame<'frame, 'buffer>,
    ) -> Option<&'b mut GlowFrame<'frame, 'buffer>> {
        Some(frame.as_mut())
    }
    fn tex_from_gl(
        context: &ContextId<GlesTexture>,
        texture: GlesTexture,
    ) -> Option<Self::TextureId> {
        Some(
            MultiTexture::from_native_texture::<GbmGlowBackend<DrmDeviceFd>>(context, texture)
                .unwrap(),
        )
    }
    fn tex_to_gl(
        context: &ContextId<GlesTexture>,
        texture: &Self::TextureId,
    ) -> Option<GlesTexture> {
        texture.get::<GbmGlowBackend<DrmDeviceFd>>(context)
    }
    fn from_gles_error(err: GlesError) -> Self::Error {
        GlMultiError::Render(err)
    }

    fn bind_glow_texture<'a>(
        &mut self,
        target: &'a mut GlesTexture,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind(target)
    }
    fn bind_glow_renderbuffer<'a>(
        &mut self,
        target: &'a mut GlesRenderbuffer,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        self.bind(target)
    }
    fn create_glow_renderbuffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoords>,
    ) -> Result<GlesRenderbuffer, Self::Error> {
        Offscreen::<GlesRenderbuffer>::create_buffer(self.as_mut(), format, size)
            .map_err(GlMultiError::Render)
    }
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        Blit::blit(self, from, to, src, dst, filter)
    }
}

impl AsGlowRenderer for super::VulkanMultiRenderer<'_> {
    fn glow_renderer(&self) -> Option<&GlowRenderer> {
        None
    }
    fn glow_renderer_mut(&mut self) -> Option<&mut GlowRenderer> {
        None
    }
    fn glow_frame<'b, 'frame, 'buffer>(
        _frame: &'b Self::Frame<'frame, 'buffer>,
    ) -> Option<&'b GlowFrame<'frame, 'buffer>> {
        None
    }
    fn glow_frame_mut<'b, 'frame, 'buffer>(
        _frame: &'b mut Self::Frame<'frame, 'buffer>,
    ) -> Option<&'b mut GlowFrame<'frame, 'buffer>> {
        None
    }
    fn tex_from_gl(
        _context: &ContextId<GlesTexture>,
        _texture: GlesTexture,
    ) -> Option<Self::TextureId> {
        None
    }
    fn tex_to_gl(
        _context: &ContextId<GlesTexture>,
        _texture: &Self::TextureId,
    ) -> Option<GlesTexture> {
        None
    }
    fn from_gles_error(err: GlesError) -> Self::Error {
        VulkanMultiError::Render(smithay::backend::renderer::vulkan::Error::GlesError(
            err.to_string(),
        ))
    }

    fn bind_glow_texture<'a>(
        &mut self,
        _target: &'a mut GlesTexture,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        Err(VulkanMultiError::Render(
            smithay::backend::renderer::vulkan::Error::UnsupportedPixelFormat,
        ))
    }
    fn bind_glow_renderbuffer<'a>(
        &mut self,
        _target: &'a mut GlesRenderbuffer,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        Err(VulkanMultiError::Render(
            smithay::backend::renderer::vulkan::Error::UnsupportedPixelFormat,
        ))
    }
    fn create_glow_renderbuffer(
        &mut self,
        _format: Fourcc,
        _size: Size<i32, BufferCoords>,
    ) -> Result<GlesRenderbuffer, Self::Error> {
        Err(VulkanMultiError::Render(
            smithay::backend::renderer::vulkan::Error::UnsupportedPixelFormat,
        ))
    }
    fn blit(
        &mut self,
        from: &Self::Framebuffer<'_>,
        to: &mut Self::Framebuffer<'_>,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, Self::Error> {
        Blit::blit(self, from, to, src, dst, filter)
    }
}

pub struct DamageElement {
    id: Id,
    geometry: Rectangle<i32, Logical>,
}

impl DamageElement {
    pub fn new(geometry: Rectangle<i32, Logical>) -> DamageElement {
        DamageElement {
            id: Id::new(),
            geometry,
        }
    }
}

impl Element for DamageElement {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        CommitCounter::default()
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size((1.0, 1.0).into())
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.geometry.to_f64().to_physical(scale).to_i32_round()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        _commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        DamageSet::from_slice(&[Rectangle::from_size(self.geometry(scale).size)])
    }
}

impl<R: Renderer> RenderElement<R> for DamageElement {
    fn draw(
        &self,
        _frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        Ok(())
    }
}
