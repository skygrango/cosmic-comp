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
        use smithay::backend::renderer::vulkan::Error as VulkanRendererError;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};

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
                assert!(
                    tested_any,
                    "At least one Vulkan 1.3 physical device should be present on this test environment"
                );
            }
        }
    }

    #[test]
    fn test_vulkan_drm_node_matching_and_formats() {
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::renderer::ImportDma;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};

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
                let Ok(file) = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(dev_path)
                else {
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

                let matches = phd
                    .render_node()
                    .ok()
                    .flatten()
                    .is_some_and(|n| n == node || Some(n) == dev_render)
                    || phd
                        .primary_node()
                        .ok()
                        .flatten()
                        .is_some_and(|n| n == node || Some(n) == dev_primary);

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
                    assert!(
                        formats
                            .iter()
                            .any(|f| f.code == smithay::backend::allocator::Fourcc::Xrgb8888),
                        "dmabuf_formats must contain Xrgb8888 for Wayland SDR client compatibility"
                    );
                    assert!(
                        formats
                            .iter()
                            .any(|f| f.code == smithay::backend::allocator::Fourcc::Xbgr8888),
                        "dmabuf_formats must contain Xbgr8888 for Wayland SDR client compatibility"
                    );
                    assert!(
                        formats
                            .iter()
                            .any(|f| f.code == smithay::backend::allocator::Fourcc::Xbgr2101010),
                        "dmabuf_formats must contain Xbgr2101010 for Wayland HDR client compatibility"
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
        use crate::backend::render::element::AsGlowRenderer;
        use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::renderer::multigpu::GpuManager;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        let phds: Vec<_> = PhysicalDevice::enumerate(&instance).unwrap().collect();
        for phd in &phds {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(file) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/dri/card1")
            else {
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

            let node = phd
                .render_node()
                .ok()
                .flatten()
                .unwrap_or_else(|| DrmNode::from_file(&drm_fd).unwrap());

            let mut backend = GbmVulkanBackend::new();
            backend.add_node(
                node,
                GbmAllocator::new(gbm, GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT),
                renderer,
            );

            let mut manager = GpuManager::new(backend)
                .expect("GpuManager should initialize with GbmVulkanBackend");
            let mut single = manager
                .single_renderer(&node)
                .expect("single_renderer should succeed for added node");

            // Test AsGlowRenderer trait methods on VulkanMultiRenderer to ensure safety
            assert!(
                single.glow_renderer().is_none(),
                "glow_renderer must be None for VulkanMultiRenderer"
            );
            assert!(
                single.glow_renderer_mut().is_none(),
                "glow_renderer_mut must be None for VulkanMultiRenderer"
            );
            assert!(
                single
                    .create_glow_renderbuffer(
                        smithay::backend::allocator::Fourcc::Argb8888,
                        (100, 100).into()
                    )
                    .is_err(),
                "create_glow_renderbuffer must return Err for VulkanMultiRenderer"
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_dmabuf_import_and_render() {
        use smithay::backend::allocator::Allocator;
        use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
        use smithay::backend::drm::DrmDeviceFd;
        use smithay::backend::renderer::multigpu::GpuManager;
        use smithay::backend::renderer::{Bind, Frame, ImportDma, Renderer};
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::utils::Transform;

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }

            let Ok(file) = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/dri/card1")
            else {
                continue;
            };
            let owned: rustix::fd::OwnedFd = file.into();
            let drm_fd = DrmDeviceFd::new(smithay::utils::DeviceFd::from(owned));
            let Ok(gbm) = GbmDevice::new(drm_fd.clone()) else {
                continue;
            };

            let Ok(renderer) = VulkanRenderer::new(&phd, Some(drm_fd.clone())) else {
                continue;
            };

            let node = phd
                .render_node()
                .ok()
                .flatten()
                .unwrap_or_else(|| DrmNode::from_file(&drm_fd).unwrap());

            println!("Device node: {:?}", node);
            let formats = renderer.dmabuf_formats();
            println!("Supported dmabuf formats count: {}", formats.iter().count());
            for f in formats
                .iter()
                .filter(|f| f.code == smithay::backend::allocator::Fourcc::Argb8888)
            {
                let plane_count = renderer
                    .device()
                    .formats()
                    .find(|e| e.format == *f)
                    .map(|e| e.modifier_properties.drm_format_modifier_plane_count)
                    .unwrap_or(0);
                println!(
                    "  Argb8888 modifier: {:?}, plane_count: {}",
                    f.modifier, plane_count
                );
            }

            let mut gbm_allocator = GbmAllocator::new(
                gbm.clone(),
                GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
            );
            let mut backend = GbmVulkanBackend::new();
            backend.add_node(
                node,
                GbmAllocator::new(gbm, GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT),
                renderer,
            );

            let mut manager = GpuManager::new(backend).expect("GpuManager init");
            let mut single = manager.single_renderer(&node).expect("single_renderer");

            // Allocate a client dmabuf
            use smithay::backend::allocator::{Buffer, dmabuf::AsDmabuf};
            let client_gbm = gbm_allocator
                .create_buffer(
                    64,
                    64,
                    smithay::backend::allocator::Fourcc::Argb8888,
                    &[smithay::backend::allocator::Modifier::Unrecognized(
                        144115188757872388,
                    )],
                )
                .expect("Failed to allocate client dmabuf with modifier 144115188757872388");
            let client_dmabuf = client_gbm.export().expect("export client dmabuf");
            println!(
                "client_dmabuf allocation: (format: {:?}, node: {:?})",
                client_dmabuf.format(),
                client_dmabuf.node()
            );

            // Try importing client_dmabuf directly into single renderer
            let imported_tex = single.import_dmabuf(&client_dmabuf, None);
            println!(
                "single.import_dmabuf result: {:?}",
                imported_tex.as_ref().map(|_| ())
            );
            if let Err(ref e) = imported_tex {
                println!("import error: {:?}", e);
            }
            assert!(
                imported_tex.is_ok(),
                "import_dmabuf on single renderer failed: {:?}",
                imported_tex.err()
            );
            let multi_tex = imported_tex.unwrap();

            // Try rendering it onto a framebuffer
            let target_gbm = gbm_allocator
                .create_buffer(
                    64,
                    64,
                    smithay::backend::allocator::Fourcc::Argb8888,
                    &[
                        smithay::backend::allocator::Modifier::Linear,
                        smithay::backend::allocator::Modifier::Invalid,
                    ],
                )
                .expect("Failed to allocate target dmabuf");
            let mut target_dmabuf = target_gbm.export().expect("export target dmabuf");

            let mut fb = single
                .bind(&mut target_dmabuf)
                .expect("Failed to bind target dmabuf");
            let mut frame = single
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("render frame");

            let render_res = frame.render_texture_from_to(
                &multi_tex,
                smithay::utils::Rectangle::new((0.0, 0.0).into(), (64.0, 64.0).into()),
                smithay::utils::Rectangle::new((0, 0).into(), (64, 64).into()),
                &[smithay::utils::Rectangle::new(
                    (0, 0).into(),
                    (64, 64).into(),
                )],
                &[],
                Transform::Normal,
                1.0,
            );
            println!("frame.render_texture_from_to result: {:?}", render_res);
            assert!(render_res.is_ok());
            let sync = frame.finish().expect("frame.finish failed");
            println!("frame.finish sync: {:?}", sync);
            let _ = sync.wait();
            let _ = single.cleanup_texture_cache();
            break;
        }
    }

    #[test]
    fn test_vulkan_host_image_copy_and_shm_import() {
        use smithay::backend::renderer::ImportMem;
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};

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
            println!(
                "VulkanImage::new (optimal) result: {:?}",
                img_optimal.as_ref().map(|_| ())
            );

            // Test linear tiling
            let img_linear = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk_format,
                vk::ImageUsageFlags::HOST_TRANSFER_EXT | vk::ImageUsageFlags::SAMPLED,
                true,
            );
            println!(
                "VulkanImage::new (linear) result: {:?}",
                img_linear.as_ref().map(|_| ())
            );

            let pixel_data = vec![255u8; 64 * 64 * 4];
            let res_argb = renderer.import_memory(
                &pixel_data,
                smithay::backend::allocator::Fourcc::Argb8888,
                (64, 64).into(),
                false,
            );
            assert!(
                res_argb.is_ok(),
                "import_memory Argb8888 should succeed: {:?}",
                res_argb.err()
            );
            assert_eq!(
                smithay::backend::renderer::Texture::format(&res_argb.unwrap()),
                Some(smithay::backend::allocator::Fourcc::Argb8888)
            );

            let res_xrgb = renderer.import_memory(
                &pixel_data,
                smithay::backend::allocator::Fourcc::Xrgb8888,
                (64, 64).into(),
                false,
            );
            assert!(
                res_xrgb.is_ok(),
                "import_memory Xrgb8888 should succeed: {:?}",
                res_xrgb.err()
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_render_clear_and_texture() {
        use smithay::backend::renderer::{Bind, Color32F, Frame, ImportMem, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage to framebuffer");

            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(32 * 32);
            let tex = renderer
                .import_memory(
                    &green_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (32, 32).into(),
                    false,
                )
                .expect("Failed to import green texture");

            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
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
            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr8888,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            println!("First 16 bytes of pixels: {:?}", &pixels[..16]);
            println!(
                "Pixels at (20, 20): {:?}",
                &pixels[(20 * 64 + 20) * 4..(20 * 64 + 20) * 4 + 4]
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_render_ab30() {
        use smithay::backend::renderer::{Bind, Color32F, Frame, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage AB30");

            use smithay::backend::renderer::ImportMem;
            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(32 * 32);
            let tex = renderer
                .import_memory(
                    &green_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (32, 32).into(),
                    false,
                )
                .expect("Failed to import green texture");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage to framebuffer");
            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
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
            println!(
                "AB30 frame.render_texture_from_to result: {:?}",
                render_tex_res
            );
            assert!(render_tex_res.is_ok());

            let sync = frame.finish().expect("frame.finish failed");
            println!("AB30 frame.finish sync: {:?}", sync);

            use smithay::backend::renderer::ExportMem;
            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            println!("AB30 First 16 bytes of pixels: {:?}", &pixels[..16]);
            let u32_val = u32::from_ne_bytes(pixels[0..4].try_into().unwrap());
            println!("AB30 First pixel as u32: 0x{:08X}", u32_val);
            println!("  R (bits 0..9):   {}", u32_val & 0x3FF);
            println!("  G (bits 10..19): {}", (u32_val >> 10) & 0x3FF);
            println!("  B (bits 20..29): {}", (u32_val >> 20) & 0x3FF);
            println!("  A (bits 30..31): {}", (u32_val >> 30) & 0x3);

            let center_idx = (20 * 64 + 20) * 4;
            let center_val =
                u32::from_ne_bytes(pixels[center_idx..center_idx + 4].try_into().unwrap());
            println!("AB30 Center pixel (20, 20) as u32: 0x{:08X}", center_val);
            println!("  Center R (bits 0..9):   {}", center_val & 0x3FF);
            println!("  Center G (bits 10..19): {}", (center_val >> 10) & 0x3FF);
            println!("  Center B (bits 20..29): {}", (center_val >> 20) & 0x3FF);
            println!("  Center A (bits 30..31): {}", (center_val >> 30) & 0x3);
            break;
        }
    }

    #[test]
    fn test_vulkan_render_ab30_hdr() {
        use smithay::backend::renderer::{Bind, Color32F, Frame, HdrOutputConfig, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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

            renderer.set_hdr_output(Some(HdrOutputConfig::default()));

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage AB30 HDR");

            use smithay::backend::renderer::ImportMem;
            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(32 * 32);
            let tex = renderer
                .import_memory(
                    &green_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (32, 32).into(),
                    false,
                )
                .expect("Failed to import green texture");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage to framebuffer");
            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            let clear_res = frame.clear(
                Color32F::new(0.153, 0.161, 0.165, 1.0),
                &[Rectangle::new((0, 0).into(), (64, 64).into())],
            );
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
            assert!(render_tex_res.is_ok());

            let sync = frame.finish().expect("frame.finish failed");

            use smithay::backend::renderer::ExportMem;
            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            let u32_val = u32::from_ne_bytes(pixels[0..4].try_into().unwrap());
            println!("HDR AB30 Clear pixel as u32: 0x{:08X}", u32_val);
            println!("  HDR Clear R (bits 0..9):   {}", u32_val & 0x3FF);
            println!("  HDR Clear G (bits 10..19): {}", (u32_val >> 10) & 0x3FF);
            println!("  HDR Clear B (bits 20..29): {}", (u32_val >> 20) & 0x3FF);
            println!("  HDR Clear A (bits 30..31): {}", (u32_val >> 30) & 0x3);

            let center_idx = (20 * 64 + 20) * 4;
            let center_val =
                u32::from_ne_bytes(pixels[center_idx..center_idx + 4].try_into().unwrap());
            println!(
                "HDR AB30 Center pixel (20, 20) as u32: 0x{:08X}",
                center_val
            );
            println!("  HDR Center R (bits 0..9):   {}", center_val & 0x3FF);
            println!(
                "  HDR Center G (bits 10..19): {}",
                (center_val >> 10) & 0x3FF
            );
            println!(
                "  HDR Center B (bits 20..29): {}",
                (center_val >> 20) & 0x3FF
            );
            println!("  HDR Center A (bits 30..31): {}", (center_val >> 30) & 0x3);

            assert_ne!(center_val, 0, "HDR Center pixel must not be 0/black!");
            assert_ne!(u32_val, 0, "HDR Clear pixel must not be 0/black!");
            break;
        }
    }

    #[test]
    fn test_vulkan_render_multi_element_desktop() {
        use smithay::backend::renderer::{Bind, Color32F, ExportMem, Frame, ImportMem, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage");

            // Wallpaper texture: 64x64 solid Blue (R:0, G:0, B:255, A:255)
            let blue_pixels = vec![0u8, 0u8, 255u8, 255u8].repeat(64 * 64);
            let wallpaper_tex = renderer
                .import_memory(
                    &blue_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (64, 64).into(),
                    false,
                )
                .expect("Failed to import wallpaper texture");

            // Window surface texture: 32x32 solid Red (R:255, G:0, B:0, A:255)
            let red_pixels = vec![255u8, 0u8, 0u8, 255u8].repeat(32 * 32);
            let window_tex = renderer
                .import_memory(
                    &red_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (32, 32).into(),
                    false,
                )
                .expect("Failed to import window texture");

            // Cursor texture: 8x8 solid Green (R:0, G:255, B:0, A:255)
            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(8 * 8);
            let cursor_tex = renderer
                .import_memory(
                    &green_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (8, 8).into(),
                    false,
                )
                .expect("Failed to import cursor texture");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage");
            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            // 1. Initial clear to black
            frame
                .clear(
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                    &[Rectangle::new((0, 0).into(), (64, 64).into())],
                )
                .expect("frame.clear failed");

            // 2. Draw wallpaper (full screen)
            frame
                .render_texture_from_to(
                    &wallpaper_tex,
                    Rectangle::new((0.0, 0.0).into(), (64.0, 64.0).into()),
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    &[Rectangle::new((0, 0).into(), (64, 64).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render wallpaper failed");

            // 3. Draw window surface at (16, 16)
            frame
                .render_texture_from_to(
                    &window_tex,
                    Rectangle::new((0.0, 0.0).into(), (32.0, 32.0).into()),
                    Rectangle::new((16, 16).into(), (32, 32).into()),
                    &[Rectangle::new((0, 0).into(), (32, 32).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render window failed");

            // 4. Draw cursor at (40, 40)
            frame
                .render_texture_from_to(
                    &cursor_tex,
                    Rectangle::new((0.0, 0.0).into(), (8.0, 8.0).into()),
                    Rectangle::new((40, 40).into(), (8, 8).into()),
                    &[Rectangle::new((0, 0).into(), (8, 8).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render cursor failed");

            let sync = frame.finish().expect("frame.finish failed");
            println!("Multi-element desktop sync: {:?}", sync);

            // Read back the framebuffer
            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr8888,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            // Pixel at (4, 4) should be Wallpaper: Blue [0, 0, 255, 255]
            let wp_pixel = &pixels[(4 * 64 + 4) * 4..(4 * 64 + 4) * 4 + 4];
            println!("Wallpaper pixel at (4, 4): {:?}", wp_pixel);
            assert_eq!(
                wp_pixel,
                &[0, 0, 255, 255],
                "Wallpaper should be blue and not overwritten/undefined"
            );

            // Pixel at (20, 20) should be Window: Red [255, 0, 0, 255]
            let win_pixel = &pixels[(20 * 64 + 20) * 4..(20 * 64 + 20) * 4 + 4];
            println!("Window pixel at (20, 20): {:?}", win_pixel);
            assert_eq!(win_pixel, &[255, 0, 0, 255], "Window should be red");

            // Pixel at (42, 42) should be Cursor: Green [0, 255, 0, 255]
            let cur_pixel = &pixels[(42 * 64 + 42) * 4..(42 * 64 + 42) * 4 + 4];
            println!("Cursor pixel at (42, 42): {:?}", cur_pixel);
            assert_eq!(cur_pixel, &[0, 255, 0, 255], "Cursor should be green");

            println!("Multi-element desktop test PASSED!");
            break;
        }
    }

    #[test]
    fn test_vulkan_xrgb_cursor_trail_prevention() {
        use smithay::backend::renderer::{Bind, Color32F, ExportMem, Frame, ImportMem, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target image");

            // 1. Cursor texture: solid Green [0, 255, 0, 255]
            let green_pixels = vec![0u8, 255u8, 0u8, 255u8].repeat(16 * 16);
            let cursor_tex = renderer
                .import_memory(
                    &green_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import cursor texture");

            // 2. XBGR8888 background buffer where unused X byte is explicitly 0x00!
            // Format: [R: 0, G: 0, B: 255, X: 0]
            let xbgr_pixels = vec![0u8, 0u8, 255u8, 0u8].repeat(16 * 16);
            let xbgr_tex = renderer
                .import_memory(
                    &xbgr_pixels,
                    smithay::backend::allocator::Fourcc::Xbgr8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import XBGR texture");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage");
            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            // First: draw the cursor at (16, 16)
            frame
                .render_texture_from_to(
                    &cursor_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((16, 16).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render cursor failed");

            // Second: redraw the damaged area with XBGR texture (wallpaper/window redrawing over old cursor)
            frame
                .render_texture_from_to(
                    &xbgr_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((16, 16).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render XBGR texture over old cursor failed");

            let _ = frame.finish().expect("frame.finish failed");

            // Read back pixel at (20, 20): must be Blue [0, 0, 255, 255], NOT Green!
            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr8888,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");
            let pixel = &pixels[(20 * 64 + 20) * 4..(20 * 64 + 20) * 4 + 4];
            println!("Overwritten pixel at (20, 20): {:?}", pixel);
            assert_eq!(
                pixel,
                &[0, 0, 255, 255],
                "Old cursor must be completely erased by XBGR buffer (no cursor trails!)"
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_hdr_linear_blending() {
        use smithay::backend::renderer::{
            Bind, Color32F, ExportMem, Frame, HdrOutputConfig, ImportMem, Renderer,
        };
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
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

            renderer.set_hdr_output(Some(HdrOutputConfig::default()));

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target image");

            // Solid White texture [255, 255, 255, 255]
            let white_pixels = vec![255u8, 255u8, 255u8, 255u8].repeat(32 * 32);
            let white_tex = renderer
                .import_memory(
                    &white_pixels,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (32, 32).into(),
                    false,
                )
                .expect("Failed to import white texture");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage");
            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            // 1. Clear to black (0 nits)
            frame
                .clear(
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                    &[Rectangle::new((0, 0).into(), (64, 64).into())],
                )
                .expect("frame.clear failed");

            // 2. Render white texture with alpha = 0.5 (semi-transparent)
            // 50% opacity of 203 nits white in linear space is 101.5 nits.
            // In ST 2084 (PQ), 101.5 nits is ~0.5098.
            // On a 10-bit integer scale (0..1023): 0.5098 * 1023 ≈ 521.
            // If it had incorrectly blended in non-linear PQ space:
            // 0.5 * 0.5806 (203 nits in PQ) = 0.2903 in PQ, which in 10-bit is 297 (5.4 nits!).
            frame
                .render_texture_from_to(
                    &white_tex,
                    Rectangle::new((0.0, 0.0).into(), (32.0, 32.0).into()),
                    Rectangle::new((16, 16).into(), (32, 32).into()),
                    &[Rectangle::new((0, 0).into(), (32, 32).into())],
                    &[],
                    Transform::Normal,
                    0.5,
                )
                .expect("render_texture_from_to failed");

            let _ = frame.finish().expect("frame.finish failed");

            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            let center_idx = (20 * 64 + 20) * 4;
            let center_val =
                u32::from_ne_bytes(pixels[center_idx..center_idx + 4].try_into().unwrap());
            let r = center_val & 0x3FF;
            let g = (center_val >> 10) & 0x3FF;
            let b = (center_val >> 20) & 0x3FF;
            println!("HDR 50% blended pixel (20, 20): R={}, G={}, B={}", r, g, b);

            // Verify linear blending: values should be around 510-530, NOT crushed to ~290!
            assert!(
                r > 480 && r < 560,
                "R must be around ~521 (linear ~101.5 nits), got {}",
                r
            );
            assert!(
                g > 480 && g < 560,
                "G must be around ~521 (linear ~101.5 nits), got {}",
                g
            );
            assert!(
                b > 480 && b < 560,
                "B must be around ~521 (linear ~101.5 nits), got {}",
                b
            );
            println!("HDR Linear Blending test PASSED!");
            break;
        }
    }

    #[test]
    fn test_vulkan_hdr_passthrough_and_color_transform() {
        use smithay::backend::renderer::color::HdrOutputConfig;
        use smithay::backend::renderer::{Bind, Color32F, ExportMem, Frame, ImportMem, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::utils::{Rectangle, Transform};
        use smithay::wayland::color::management::ImageDescription;

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }
            let mut renderer = match VulkanRenderer::new(&phd, None) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let config = HdrOutputConfig {
                reference_white: 203.0,
                max_luminance: 1000.0,
                sdr_gamma: 0.0,
                gamut_stretch: 0.0,
                hardware_offload: false,
                is_sdr: false,
            };
            renderer.set_hdr_output(Some(config));

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage to framebuffer");

            // Test 1: Tagged HDR PQ surface (WINDOWS_BT2100) -> passthrough!
            let pq_shm_data = vec![148u8, 148u8, 148u8, 255u8].repeat(16 * 16);
            let hdr_tex = renderer
                .import_memory(
                    &pq_shm_data,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import HDR texture");

            // Test 2: Tagged SDR surface (SRGB) pure white (255, 255, 255) -> converted to 203 nits PQ (~594)!
            let sdr_white_data = vec![255u8, 255u8, 255u8, 255u8].repeat(16 * 16);
            let sdr_tex = renderer
                .import_memory(
                    &sdr_white_data,
                    smithay::backend::allocator::Fourcc::Abgr8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import SDR texture");

            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            frame
                .clear(
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                    &[Rectangle::new((0, 0).into(), (64, 64).into())],
                )
                .unwrap();

            frame.set_surface_color_description(Some(&ImageDescription::WINDOWS_BT2100));
            frame
                .render_texture_from_to(
                    &hdr_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((0, 0).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render_texture_from_to failed");
            frame.set_surface_color_description(None);

            frame.set_surface_color_description(Some(&ImageDescription::SRGB));
            frame
                .render_texture_from_to(
                    &sdr_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((32, 32).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render_texture_from_to failed");
            frame.set_surface_color_description(None);

            let _ = frame.finish().expect("frame.finish failed");

            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            // Check Test 1 (passthrough): pixel at (8, 8)
            let p1_idx = (8 * 64 + 8) * 4;
            let p1_val = u32::from_ne_bytes(pixels[p1_idx..p1_idx + 4].try_into().unwrap());
            let r1 = p1_val & 0x3FF;
            let g1 = (p1_val >> 10) & 0x3FF;
            let b1 = (p1_val >> 20) & 0x3FF;
            println!(
                "HDR PQ Passthrough pixel (8, 8): R={}, G={}, B={}",
                r1, g1, b1
            );
            assert!(
                (r1 as i32 - 594).abs() <= 2,
                "Passthrough R must be ~594, got {}",
                r1
            );
            assert!(
                (g1 as i32 - 594).abs() <= 2,
                "Passthrough G must be ~594, got {}",
                g1
            );
            assert!(
                (b1 as i32 - 594).abs() <= 2,
                "Passthrough B must be ~594, got {}",
                b1
            );

            // Check Test 2 (SDR -> HDR): pixel at (40, 40)
            let p2_idx = (40 * 64 + 40) * 4;
            let p2_val = u32::from_ne_bytes(pixels[p2_idx..p2_idx + 4].try_into().unwrap());
            let r2 = p2_val & 0x3FF;
            let g2 = (p2_val >> 10) & 0x3FF;
            let b2 = (p2_val >> 20) & 0x3FF;
            println!(
                "SDR->HDR Converted pixel (40, 40): R={}, G={}, B={}",
                r2, g2, b2
            );
            assert!(
                (r2 as i32 - 594).abs() <= 2,
                "SDR->HDR R must be ~594 (203 nits PQ), got {}",
                r2
            );
            assert!(
                (g2 as i32 - 594).abs() <= 2,
                "SDR->HDR G must be ~594 (203 nits PQ), got {}",
                g2
            );
            assert!(
                (b2 as i32 - 594).abs() <= 2,
                "SDR->HDR B must be ~594 (203 nits PQ), got {}",
                b2
            );

            println!("HDR Passthrough and Color Transform test PASSED!");
            break;
        }
    }

    #[test]
    fn test_vulkan_hdr_xrgb_wallpaper_and_panel() {
        use smithay::backend::renderer::color::HdrOutputConfig;
        use smithay::backend::renderer::{Bind, Color32F, ExportMem, Frame, ImportMem, Renderer};
        use smithay::backend::vulkan::ash::vk;
        use smithay::backend::vulkan::image::VulkanImage;
        use smithay::backend::vulkan::{Instance, PhysicalDevice, version::Version};
        use smithay::utils::{Rectangle, Transform};

        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };

        for phd in PhysicalDevice::enumerate(&instance).unwrap() {
            if phd.api_version() < Version::VERSION_1_3 {
                continue;
            }
            let mut renderer = match VulkanRenderer::new(&phd, None) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let config = HdrOutputConfig {
                reference_white: 203.0,
                max_luminance: 1000.0,
                sdr_gamma: 2.2,
                gamut_stretch: 0.0,
                hardware_offload: false,
                is_sdr: false,
            };
            renderer.set_hdr_output(Some(config));

            let mut target_image = VulkanImage::new(
                renderer.device(),
                64,
                64,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
                false,
            )
            .expect("Failed to create target VulkanImage");

            let mut fb = renderer
                .bind(&mut target_image)
                .expect("Failed to bind VulkanImage to framebuffer");

            // Wallpaper: cosmic-bg uses Xrgb8888 where byte 0=B, 1=G, 2=R, 3=0x00
            // Test 1: Pure Red in Xrgb8888 -> [0, 0, 255, 0]
            let red_xrgb = vec![0u8, 0u8, 255u8, 0u8].repeat(16 * 16);
            let red_wp_tex = renderer
                .import_memory(
                    &red_xrgb,
                    smithay::backend::allocator::Fourcc::Xrgb8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import red Xrgb8888 texture");

            // Test 2: Pure Blue in Xrgb8888 -> [255, 0, 0, 0]
            let blue_xrgb = vec![255u8, 0u8, 0u8, 0u8].repeat(16 * 16);
            let blue_wp_tex = renderer
                .import_memory(
                    &blue_xrgb,
                    smithay::backend::allocator::Fourcc::Xrgb8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import blue Xrgb8888 texture");

            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            frame
                .clear(
                    Color32F::new(0.0, 0.0, 0.0, 1.0),
                    &[Rectangle::new((0, 0).into(), (64, 64).into())],
                )
                .unwrap();

            // Render Red wallpaper at (0, 0)
            frame
                .render_texture_from_to(
                    &red_wp_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((0, 0).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render red wallpaper failed");

            // Render Blue wallpaper at (32, 0)
            frame
                .render_texture_from_to(
                    &blue_wp_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((32, 0).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render blue wallpaper failed");

            let _ = frame.finish().expect("frame.finish failed");

            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            // Read Red wallpaper at (8, 8)
            let r_idx = (8 * 64 + 8) * 4;
            let r_val = u32::from_ne_bytes(pixels[r_idx..r_idx + 4].try_into().unwrap());
            let r_r = r_val & 0x3FF;
            let r_g = (r_val >> 10) & 0x3FF;
            let r_b = (r_val >> 20) & 0x3FF;
            println!("Red Xrgb8888 pixel (8, 8): R={}, G={}, B={}", r_r, r_g, r_b);

            // Read Blue wallpaper at (8, 40)
            let b_idx = (8 * 64 + 40) * 4;
            let b_val = u32::from_ne_bytes(pixels[b_idx..b_idx + 4].try_into().unwrap());
            let b_r = b_val & 0x3FF;
            let b_g = (b_val >> 10) & 0x3FF;
            let b_b = (b_val >> 20) & 0x3FF;
            println!(
                "Blue Xrgb8888 pixel (8, 40): R={}, G={}, B={}",
                b_r, b_g, b_b
            );

            // In BT.2020 PQ:
            // Red (1.0, 0.0, 0.0) -> R should be highest (~540) and B lowest (~208)
            // If R and B are swapped, r_b would be higher than r_r!
            assert!(
                r_r > r_b,
                "Red XRGB must have R > B! got R={}, B={}",
                r_r,
                r_b
            );
            assert!(
                b_b > b_r,
                "Blue XRGB must have B > R! got R={}, B={}",
                b_r,
                b_b
            );

            // Test 3: cosmic-panel uses Argb8888 dmabuf/shm
            // Panel button: pure Red in Argb8888 -> [0, 0, 255, 255]
            let red_argb = vec![0u8, 0u8, 255u8, 255u8].repeat(16 * 16);
            let red_panel_tex = renderer
                .import_memory(
                    &red_argb,
                    smithay::backend::allocator::Fourcc::Argb8888,
                    (16, 16).into(),
                    false,
                )
                .expect("Failed to import red Argb8888 texture");

            let mut frame = renderer
                .render(&mut fb, (64, 64).into(), Transform::Normal)
                .expect("Failed to create frame");

            frame
                .render_texture_from_to(
                    &red_panel_tex,
                    Rectangle::new((0.0, 0.0).into(), (16.0, 16.0).into()),
                    Rectangle::new((48, 0).into(), (16, 16).into()),
                    &[Rectangle::new((0, 0).into(), (16, 16).into())],
                    &[],
                    Transform::Normal,
                    1.0,
                )
                .expect("render red panel failed");

            let _ = frame.finish().expect("frame.finish failed");

            let mapping = renderer
                .copy_framebuffer(
                    &fb,
                    Rectangle::new((0, 0).into(), (64, 64).into()),
                    smithay::backend::allocator::Fourcc::Abgr2101010,
                )
                .expect("copy_framebuffer failed");
            let pixels = renderer.map_texture(&mapping).expect("map_texture failed");

            let panel_idx = (8 * 64 + 56) * 4;
            let panel_val =
                u32::from_ne_bytes(pixels[panel_idx..panel_idx + 4].try_into().unwrap());
            let panel_r = panel_val & 0x3FF;
            let panel_g = (panel_val >> 10) & 0x3FF;
            let panel_b = (panel_val >> 20) & 0x3FF;
            println!(
                "Red Argb8888 Panel pixel (8, 56): R={}, G={}, B={}",
                panel_r, panel_g, panel_b
            );
            assert!(
                panel_r > panel_b,
                "Red ARGB panel must have R > B! got R={}, B={}",
                panel_r,
                panel_b
            );
            break;
        }
    }

    #[test]
    fn test_vulkan_software_renderer_offscreen_fallback() {
        let renderer_res = crate::backend::kms::software_renderer();
        if let Ok(mut renderer) = renderer_res {
            let size = smithay::utils::Size::from((100, 100));
            let mut ref_renderer = crate::backend::render::RendererRef::Glow(&mut renderer);
            let constraints =
                crate::wayland::handlers::image_copy_capture::constraints_for_renderer(
                    size,
                    &mut ref_renderer,
                );
            assert_eq!(constraints.size, size);
            assert!(
                constraints.shm.contains(
                    &smithay::reexports::wayland_server::protocol::wl_shm::Format::Abgr8888
                )
            );
        }
    }
}
