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
    use smithay::backend::renderer::{
        multigpu::MultiRenderer,
        Bind, ExportMem, ImportDma, ImportMem, Renderer,
    };
    use smithay::backend::drm::DrmDeviceFd;
    use smithay::backend::allocator::dmabuf::Dmabuf;

    use smithay::backend::renderer::ImportAll;

    fn assert_traits<'a, R>()
    where
        R: Renderer + Bind<Dmabuf> + ImportAll + ImportDma + ImportMem + ExportMem,
        R::TextureId: Clone + Send + 'static,
    {}

    fn assert_drm_output_bounds<'a, R>()
    where
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: smithay::backend::renderer::Texture + 'static,
        R::Error: std::error::Error + Send + Sync + 'static,
    {}

    #[test]
    fn test_vulkan_multi_renderer_traits() {
        assert_traits::<MultiRenderer<'static, 'static, GbmVulkanBackend<DrmDeviceFd>, GbmVulkanBackend<DrmDeviceFd>>>();
        assert_drm_output_bounds::<MultiRenderer<'static, 'static, GbmVulkanBackend<DrmDeviceFd>, GbmVulkanBackend<DrmDeviceFd>>>();
    }
}
