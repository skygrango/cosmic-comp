// SPDX-License-Identifier: GPL-3.0-only

use smithay::{
    backend::{
        SwapBuffersError,
        allocator::{
            Allocator,
            dmabuf::{AnyError, Dmabuf, DmabufAllocator},
            gbm::GbmAllocator,
        },
        drm::{CreateDrmNodeError, DrmNode},
        renderer::{
            multigpu::{ApiDevice, GraphicsApi},
            vulkan::{Error as VulkanError, VulkanRenderer},
        },
    },
    reexports::drm::control::Device,
};
use std::{
    cell::Cell,
    collections::HashMap,
    fmt,
    os::unix::prelude::AsFd,
    sync::atomic::{AtomicBool, Ordering},
};

/// Errors raised by the [`GbmVulkanBackend`]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Vulkan error
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// Error creating a drm node
    #[error(transparent)]
    DrmNode(#[from] CreateDrmNodeError),
}

impl From<Error> for SwapBuffersError {
    fn from(err: Error) -> SwapBuffersError {
        match err {
            x @ Error::DrmNode(_) => SwapBuffersError::ContextLost(Box::new(x)),
            Error::Vulkan(x) => SwapBuffersError::ContextLost(Box::new(x)),
        }
    }
}

/// A [`GraphicsApi`] utilizing user-provided GBM Devices and Vulkan for rendering.
pub struct GbmVulkanBackend<A: AsFd + 'static> {
    devices: HashMap<DrmNode, (GbmAllocator<A>, Cell<Option<VulkanRenderer>>)>,
    needs_enumeration: AtomicBool,
}

impl<A: AsFd + fmt::Debug + 'static> fmt::Debug for GbmVulkanBackend<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GbmVulkanBackend")
            .field("devices", &self.devices.keys())
            .field("needs_enumeration", &self.needs_enumeration)
            .finish()
    }
}

impl<A: AsFd + 'static> Default for GbmVulkanBackend<A> {
    fn default() -> Self {
        GbmVulkanBackend {
            devices: HashMap::new(),
            needs_enumeration: AtomicBool::new(true),
        }
    }
}

impl<A: AsFd + Clone + Send + 'static> GbmVulkanBackend<A> {
    pub fn new() -> Self {
        GbmVulkanBackend {
            devices: HashMap::new(),
            needs_enumeration: AtomicBool::new(false),
        }
    }

    pub fn current_devices(&self) -> impl Iterator<Item = &DrmNode> {
        self.devices.keys()
    }

    pub fn add_node(&mut self, node: DrmNode, gbm: GbmAllocator<A>, renderer: VulkanRenderer) {
        if self.devices.contains_key(&node) {
            return;
        }

        self.devices.insert(node, (gbm, Cell::new(Some(renderer))));
        self.needs_enumeration.store(true, Ordering::SeqCst);
    }

    /// Remove a given node from the api
    pub fn remove_node(&mut self, node: &DrmNode) {
        if self.devices.remove(node).is_some() {
            self.needs_enumeration.store(true, Ordering::SeqCst);
        }
    }
}

impl<A: AsFd + Device + Clone + 'static> GraphicsApi for GbmVulkanBackend<A> {
    type Device = GbmVulkanDevice;
    type Error = Error;

    fn enumerate(&self, list: &mut Vec<Self::Device>) -> Result<(), Self::Error> {
        self.needs_enumeration.store(false, Ordering::SeqCst);

        // remove old stuff
        list.retain(|renderer| {
            self.devices
                .keys()
                .any(|node| renderer.node.dev_id() == node.dev_id())
        });

        // add new stuff
        let new_renderers = self
            .devices
            .iter()
            // but don't replace already initialized renderers
            .filter(|(node, _)| {
                !list
                    .iter()
                    .any(|renderer| renderer.node.dev_id() == node.dev_id())
            })
            .flat_map(|(node, (allocator, renderer))| {
                let renderer = renderer.replace(None)?;
                Some(GbmVulkanDevice {
                    node: *node,
                    renderer,
                    allocator: Box::new(DmabufAllocator(allocator.clone())),
                })
            })
            .collect::<Vec<GbmVulkanDevice>>();
        list.extend(new_renderers);

        Ok(())
    }

    fn needs_enumeration(&self) -> bool {
        self.needs_enumeration.load(Ordering::Acquire)
    }

    fn identifier() -> &'static str {
        "gbm_vulkan"
    }
}

/// [`ApiDevice`] of the [`GbmVulkanBackend`]
pub struct GbmVulkanDevice {
    node: DrmNode,
    renderer: VulkanRenderer,
    allocator: Box<dyn Allocator<Buffer = Dmabuf, Error = AnyError>>,
}

impl fmt::Debug for GbmVulkanDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GbmVulkanDevice")
            .field("node", &self.node)
            .field("renderer", &self.renderer)
            .finish_non_exhaustive()
    }
}

impl ApiDevice for GbmVulkanDevice {
    type Renderer = VulkanRenderer;

    fn renderer(&self) -> &Self::Renderer {
        &self.renderer
    }
    fn renderer_mut(&mut self) -> &mut Self::Renderer {
        &mut self.renderer
    }
    fn allocator(&mut self) -> &mut dyn Allocator<Buffer = Dmabuf, Error = AnyError> {
        self.allocator.as_mut()
    }
    fn node(&self) -> &DrmNode {
        &self.node
    }
    fn can_do_cross_device_imports(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use smithay::backend::allocator::dmabuf::Dmabuf;
    use smithay::backend::drm::DrmDeviceFd;
    use smithay::backend::renderer::{
        Bind, ExportMem, ImportDma, ImportMem, Renderer, multigpu::MultiRenderer,
    };

    use smithay::backend::renderer::ImportAll;

    fn assert_traits<'a, R>()
    where
        R: Renderer + Bind<Dmabuf> + ImportAll + ImportDma + ImportMem + ExportMem,
        R::TextureId: Clone + Send + 'static,
    {
    }

    fn assert_drm_output_bounds<'a, R>()
    where
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: smithay::backend::renderer::Texture + 'static,
        R::Error: std::error::Error + Send + Sync + 'static,
    {
    }

    #[test]
    fn test_vulkan_multi_renderer_traits() {
        assert_traits::<
            MultiRenderer<
                'static,
                'static,
                GbmVulkanBackend<DrmDeviceFd>,
                GbmVulkanBackend<DrmDeviceFd>,
            >,
        >();
        assert_drm_output_bounds::<
            MultiRenderer<
                'static,
                'static,
                GbmVulkanBackend<DrmDeviceFd>,
                GbmVulkanBackend<DrmDeviceFd>,
            >,
        >();
    }

    #[test]
    fn test_vulkan_instance_version_requirement() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::renderer::vulkan::Error as VulkanRendererError;

        // 1. Verify the root cause of the startup failure:
        // When cosmic-comp was initialized with Version 1.2, phd.api_version() was clamped
        // to 1.2. VulkanRenderer requires MIN_DEVICE_VERSION = 1.3, so VulkanRenderer::new
        // returned Err(UnsupportedVersion), failing Device::new and aborting startup.
        if let Ok(instance_1_2) = Instance::new(Version::VERSION_1_2, None) {
            if let Ok(phds) = PhysicalDevice::enumerate(&instance_1_2) {
                for phd in phds {
                    let res = VulkanRenderer::new(&phd, None);
                    assert!(
                        matches!(res, Err(VulkanRendererError::UnsupportedVersion)),
                        "VulkanRenderer::new with Vulkan 1.2 instance must fail with UnsupportedVersion, but got: {:?}",
                        res.as_ref().map(|_| ())
                    );
                }
            }
        }

        // 2. Verify that initializing with Version 1.3 satisfies MIN_DEVICE_VERSION
        // and VulkanRenderer::new succeeds for capable physical devices.
        if let Ok(instance_1_3) = Instance::new(Version::VERSION_1_3, None) {
            assert!(
                instance_1_3.api_version() >= Version::VERSION_1_3,
                "Instance 1.3 must have api_version >= 1.3"
            );
            if let Ok(phds) = PhysicalDevice::enumerate(&instance_1_3) {
                let mut tested_any = false;
                for phd in phds {
                    if phd.api_version() >= Version::VERSION_1_3 {
                        let res = VulkanRenderer::new(&phd, None);
                        assert!(
                            res.is_ok(),
                            "VulkanRenderer::new with Vulkan 1.3 instance should succeed for {:?}, but got: {:?}",
                            phd.name(),
                            res.err()
                        );
                        tested_any = true;
                    }
                }
                assert!(tested_any, "At least one Vulkan 1.3 physical device should be present on this test environment");
            }
        }
    }

    #[test]
    fn test_vulkan_drm_node_matching_and_formats() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::renderer::ImportDma;

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        let phds: Vec<_> = PhysicalDevice::enumerate(&instance).unwrap().collect();
        for phd in &phds {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            // Test with both /dev/dri/card1 and /dev/dri/renderD128 if available
            for dev_path in &["/dev/dri/card1", "/dev/dri/renderD128"] {
                let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).open(dev_path) else {
                    continue;
                };
                let owned: rustix::fd::OwnedFd = file.into();
                let drm_fd = DrmDeviceFd::new(smithay::utils::DeviceFd::from(owned));
                let node = DrmNode::from_file(&drm_fd).unwrap();

                let dev_render = node
                    .node_with_type(smithay::backend::drm::NodeType::Render)
                    .and_then(|r| r.ok());
                let dev_primary = node
                    .node_with_type(smithay::backend::drm::NodeType::Primary)
                    .and_then(|r| r.ok());

                let matches = phd.render_node().ok().flatten().is_some_and(|n| n == node || Some(n) == dev_render)
                    || phd.primary_node().ok().flatten().is_some_and(|n| n == node || Some(n) == dev_primary);

                let res = VulkanRenderer::new(phd, Some(drm_fd.clone()));
                if matches {
                    assert!(
                        res.is_ok(),
                        "Matching DRM node {:?} with physical device {:?} must succeed, but got: {:?}",
                        dev_path,
                        phd.name(),
                        res.err()
                    );
                    let renderer = res.unwrap();
                    let formats = renderer.dmabuf_formats();
                    assert!(
                        formats.iter().count() > 0,
                        "dmabuf_formats on physical device {:?} should not be empty",
                        phd.name()
                    );
                } else {
                    assert!(
                        matches!(res, Err(VulkanError::MismatchedDrmDevice)),
                        "Non-matching DRM node {:?} with physical device {:?} must return MismatchedDrmDevice",
                        dev_path,
                        phd.name()
                    );
                }
            }
        }
    }

    #[test]
    fn test_gbm_vulkan_backend_and_gpu_manager() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::renderer::multigpu::GpuManager;
        use crate::backend::render::element::AsGlowRenderer;

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        let phds: Vec<_> = PhysicalDevice::enumerate(&instance).unwrap().collect();
        for phd in &phds {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/dri/card1") else {
                continue;
            };
            let owned: rustix::fd::OwnedFd = file.into();
            let drm_fd = DrmDeviceFd::new(smithay::utils::DeviceFd::from(owned));
            let Ok(gbm) = GbmDevice::new(drm_fd.clone()) else {
                continue;
            };

            let Ok(renderer) = VulkanRenderer::new(phd, Some(drm_fd.clone())) else {
                continue;
            };

            let node = phd.render_node().ok().flatten().unwrap_or_else(|| {
                DrmNode::from_file(&drm_fd).unwrap()
            });

            let mut backend = GbmVulkanBackend::new();
            backend.add_node(
                node,
                GbmAllocator::new(gbm, GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT),
                renderer,
            );

            let mut manager = GpuManager::new(backend).expect("GpuManager should initialize with GbmVulkanBackend");
            let mut single = manager.single_renderer(&node).expect("single_renderer should succeed for added node");

            // Test AsGlowRenderer trait methods on VulkanMultiRenderer to ensure safety
            assert!(single.glow_renderer().is_none(), "glow_renderer must be None for VulkanMultiRenderer");
            assert!(single.glow_renderer_mut().is_none(), "glow_renderer_mut must be None for VulkanMultiRenderer");
            assert!(
                single.create_glow_renderbuffer(smithay::backend::allocator::Fourcc::Argb8888, (100, 100).into()).is_err(),
                "create_glow_renderbuffer must return Err for VulkanMultiRenderer"
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_host_image_copy_and_shm_import() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::renderer::ImportMem;
        use smithay::backend::vulkan::ash::vk;

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            let vk_format = vk::Format::B8G8R8A8_UNORM;
            
            // Test optimal tiling
            let img_optimal = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk_format,
                vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                false,
            );
            println!("VulkanImage::new (optimal) result: {:?}", img_optimal.as_ref().map(|_| ()));

            // Test linear tiling
            let img_linear = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk_format,
                vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                true,
            );
            println!("VulkanImage::new (linear) result: {:?}", img_linear.as_ref().map(|_| ()));

            let pixel_data = vec![255u8; 64 * 64 * 4];
            let res_argb = renderer.import_memory(
                &pixel_data,
                smithay::backend::allocator::Fourcc::Argb8888,
                (64, 64).into(),
                false,
            );
            assert!(res_argb.is_ok(), "import_memory Argb8888 should succeed: {:?}", res_argb.err());
            assert_eq!(smithay::backend::renderer::Texture::format(&res_argb.unwrap()), Some(smithay::backend::allocator::Fourcc::Argb8888));

            let res_xrgb = renderer.import_memory(
                &pixel_data,
                smithay::backend::allocator::Fourcc::Xrgb8888,
                (64, 64).into(),
                false,
            );
            assert!(res_xrgb.is_ok(), "import_memory Xrgb8888 should succeed: {:?}", res_xrgb.err());
            break;
        }
    }

    #[test]
    fn test_vulkan_render_clear_and_texture() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::renderer::{Renderer, Frame, Bind, ImportMem, Color32F};
        use smithay::utils::{Rectangle, Transform};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            ).expect("Failed to create target VulkanImage");

            let mut fb = renderer.bind(&mut target_image).expect("Failed to bind VulkanImage to framebuffer");

            let green_pixels = vec![
                0u8, 255u8, 0u8, 255u8,
            ].repeat(32 * 32);
            let tex = renderer.import_memory(
                &green_pixels,
                smithay::backend::allocator::Fourcc::Abgr8888,
                (32, 32).into(),
                false,
            ).expect("Failed to import green texture");

            let mut frame = renderer.render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            // Test 1: Clear with red
            let clear_res = frame.clear(
                Color32F::new(1.0, 0.0, 0.0, 1.0),
                &[Rectangle::new((0, 0).into(), (64, 64).into())],
            );
            println!("frame.clear result: {:?}", clear_res);
            assert!(clear_res.is_ok());

            let render_tex_res = frame.render_texture_from_to(
                &tex,
                Rectangle::new((0.0, 0.0).into(), (32.0, 32.0).into()),
                Rectangle::new((16, 16).into(), (32, 32).into()),
                &[Rectangle::new((0, 0).into(), (32, 32).into())],
                &[],
                Transform::Normal,
                1.0,
            );
            println!("frame.render_texture_from_to result: {:?}", render_tex_res);
            assert!(render_tex_res.is_ok());

            let sync = frame.finish().expect("frame.finish failed");
            println!("frame.finish sync: {:?}", sync);

            use smithay::backend::renderer::ExportMem;
            let mapping = renderer.copy_framebuffer(
                &fb,
                Rectangle::new((0, 0).into(), (64, 64).into()),
                smithay::backend::allocator::Fourcc::Abgr8888,
            ).expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            println!("First 16 bytes of pixels: {:?}", &pixels[..16]);
            println!("Pixels at (20, 20): {:?}", &pixels[(20 * 64 + 20) * 4 .. (20 * 64 + 20) * 4 + 4]);
            break;
        }
    }

    #[test]
    fn test_vulkan_render_ab30() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::renderer::{Renderer, Frame, Bind, Color32F};
        use smithay::utils::{Rectangle, Transform};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            ).expect("Failed to create target VulkanImage AB30");

            use smithay::backend::renderer::ImportMem;
            let green_pixels = vec![
                0u8, 255u8, 0u8, 255u8,
            ].repeat(32 * 32);
            let tex = renderer.import_memory(
                &green_pixels,
                smithay::backend::allocator::Fourcc::Abgr8888,
                (32, 32).into(),
                false,
            ).expect("Failed to import green texture");

            let mut fb = renderer.bind(&mut target_image).expect("Failed to bind VulkanImage to framebuffer");
            let mut frame = renderer.render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            let clear_res = frame.clear(
                Color32F::new(0.153, 0.161, 0.165, 1.0),
                &[Rectangle::new((0, 0).into(), (64, 64).into())],
            );
            println!("AB30 frame.clear result: {:?}", clear_res);
            assert!(clear_res.is_ok());

            let render_tex_res = frame.render_texture_from_to(
                &tex,
                Rectangle::new((0.0, 0.0).into(), (32.0, 32.0).into()),
                Rectangle::new((16, 16).into(), (32, 32).into()),
                &[Rectangle::new((0, 0).into(), (32, 32).into())],
                &[],
                Transform::Normal,
                1.0,
            );
            println!("AB30 frame.render_texture_from_to result: {:?}", render_tex_res);
            assert!(render_tex_res.is_ok());

            let sync = frame.finish().expect("frame.finish failed");
            println!("AB30 frame.finish sync: {:?}", sync);

            use smithay::backend::renderer::ExportMem;
            let mapping = renderer.copy_framebuffer(
                &fb,
                Rectangle::new((0, 0).into(), (64, 64).into()),
                smithay::backend::allocator::Fourcc::Abgr2101010,
            ).expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            println!("AB30 First 16 bytes of pixels: {:?}", &pixels[..16]);
            let u32_val = u32::from_ne_bytes(pixels[0..4].try_into().unwrap());
            println!("AB30 First pixel as u32: 0x{:08X}", u32_val);
            println!("  R (bits 0..9):   {}", u32_val & 0x3FF);
            println!("  G (bits 10..19): {}", (u32_val >> 10) & 0x3FF);
            println!("  B (bits 20..29): {}", (u32_val >> 20) & 0x3FF);
            println!("  A (bits 30..31): {}", (u32_val >> 30) & 0x3);

            let center_idx = (20 * 64 + 20) * 4;
            let center_val = u32::from_ne_bytes(pixels[center_idx..center_idx+4].try_into().unwrap());
            println!("AB30 Center pixel (20, 20) as u32: 0x{:08X}", center_val);
            println!("  Center R (bits 0..9):   {}", center_val & 0x3FF);
            println!("  Center G (bits 10..19): {}", (center_val >> 10) & 0x3FF);
            println!("  Center B (bits 20..29): {}", (center_val >> 20) & 0x3FF);
            println!("  Center A (bits 30..31): {}", (center_val >> 30) & 0x3);
            break;
        }
    }

    #[test]
    fn test_vulkan_render_multi_element_desktop() {
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::renderer::{Renderer, Frame, Bind, Color32F, ImportMem, ExportMem};
        use smithay::utils::{Rectangle, Transform};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(mut renderer) = VulkanRenderer::new(&phd, None) else {
                continue;
            };

            // Framebuffer: 64x64 RGBA
            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            ).expect("Failed to create target VulkanImage");

            // Wallpaper texture: 64x64 solid Blue (R:0, G:0, B:255, A:255)
            let blue_pixels = vec![0u8, 0u8, 255u8, 255u8].repeat(64 * 64);
            let wallpaper_tex = renderer.import_memory(
                &blue_pixels,
                smithay::backend::allocator::Fourcc::Abgr8888,
                (64, 64).into(),
                false,
            ).expect("Failed to import wallpaper texture");

            // Window surface texture: 32x32 solid Red (R:255, G:0, B:0, A:255)
            let red_pixels = vec![255u8, 0u8, 0u8, 255u8].repeat(32 * 32);
            let window_tex = renderer.import_memory(
                &red_pixels,
                smithay::backend::allocator::Fourcc::Abgr8888,
                (32, 32).into(),
                false,
            ).expect("Failed to import window texture");

            // Cursor texture: 8x8 solid Green (R:0, G:255, B:0, A:255)
            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(8 * 8);
            let cursor_tex = renderer.import_memory(
                &green_pixels,
                smithay::backend::allocator::Fourcc::Abgr8888,
                (8, 8).into(),
                false,
            ).expect("Failed to import cursor texture");

            let mut fb = renderer.bind(&mut target_image).expect("Failed to bind VulkanImage");
            let mut frame = renderer.render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            // 1. Initial clear to black
            frame.clear(
                Color32F::new(0.0, 0.0, 0.0, 1.0),
                &[Rectangle::new((0, 0).into(), (64, 64).into())],
            ).expect("frame.clear failed");

            // 2. Draw wallpaper (full screen)
            frame.render_texture_from_to(
                &wallpaper_tex,
                Rectangle::new((0.0, 0.0).into(), (64.0, 64.0).into()),
                Rectangle::new((0, 0).into(), (64, 64).into()),
                &[Rectangle::new((0, 0).into(), (64, 64).into())],
                &[],
                Transform::Normal,
                1.0,
            ).expect("render wallpaper failed");

            // 3. Draw window surface at (16, 16)
            frame.render_texture_from_to(
                &window_tex,
                Rectangle::new((0.0, 0.0).into(), (32.0, 32.0).into()),
                Rectangle::new((16, 16).into(), (32, 32).into()),
                &[Rectangle::new((0, 0).into(), (32, 32).into())],
                &[],
                Transform::Normal,
                1.0,
            ).expect("render window failed");

            // 4. Draw cursor at (40, 40)
            frame.render_texture_from_to(
                &cursor_tex,
                Rectangle::new((0.0, 0.0).into(), (8.0, 8.0).into()),
                Rectangle::new((40, 40).into(), (8, 8).into()),
                &[Rectangle::new((0, 0).into(), (8, 8).into())],
                &[],
                Transform::Normal,
                1.0,
            ).expect("render cursor failed");

            let sync = frame.finish().expect("frame.finish failed");
            println!("Multi-element desktop sync: {:?}", sync);

            // Read back the framebuffer
            let mapping = renderer.copy_framebuffer(
                &fb,
                Rectangle::new((0, 0).into(), (64, 64).into()),
                smithay::backend::allocator::Fourcc::Abgr8888,
            ).expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            // Pixel at (4, 4) should be Wallpaper: Blue [0, 0, 255, 255]
            let wp_pixel = &pixels[(4 * 64 + 4) * 4 .. (4 * 64 + 4) * 4 + 4];
            println!("Wallpaper pixel at (4, 4): {:?}", wp_pixel);
            assert_eq!(wp_pixel, &[0, 0, 255, 255], "Wallpaper should be blue and not overwritten/undefined");

            // Pixel at (20, 20) should be Window: Red [255, 0, 0, 255]
            let win_pixel = &pixels[(20 * 64 + 20) * 4 .. (20 * 64 + 20) * 4 + 4];
            println!("Window pixel at (20, 20): {:?}", win_pixel);
            assert_eq!(win_pixel, &[255, 0, 0, 255], "Window should be red");

            // Pixel at (42, 42) should be Cursor: Green [0, 255, 0, 255]
            let cur_pixel = &pixels[(42 * 64 + 42) * 4 .. (42 * 64 + 42) * 4 + 4];
            println!("Cursor pixel at (42, 42): {:?}", cur_pixel);
            assert_eq!(cur_pixel, &[0, 255, 0, 255], "Cursor should be green");

            println!("Multi-element desktop test PASSED!");
            break;
        }
    }
}
