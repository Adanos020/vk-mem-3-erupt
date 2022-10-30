//! Easy to use, high performance memory manager for Vulkan.

#![allow(invalid_value)]

extern crate erupt;
#[macro_use]
extern crate bitflags;
#[cfg(feature = "failure")]
extern crate failure;

pub mod error;
pub mod ffi;
pub use crate::error::{Error, ErrorKind, Result};
use std::mem;
use std::sync::Arc;

/// Main allocator object
pub struct Allocator {
    /// Pointer to internal VmaAllocator instance
    internal: ffi::VmaAllocator,
    /// Vulkan device handle
    #[allow(dead_code)]
    device: Arc<erupt::DeviceLoader>,
    /// Vulkan instance handle
    #[allow(dead_code)]
    instance: Arc<erupt::InstanceLoader>,
}

// Allocator is internally thread safe unless AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED is used (then you need to add synchronization!)
unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

/// Represents custom memory pool
///
/// Fill structure `AllocatorPoolCreateInfo` and call `Allocator::create_pool` to create it.
/// Call `Allocator::destroy_pool` to destroy it.
#[derive(Debug, Clone)]
pub struct AllocatorPool {
    /// Pointer to internal VmaPool instance
    internal: ffi::VmaPool,
}

/// Construct `AllocatorPool` with default values
impl Default for AllocatorPool {
    fn default() -> Self {
        AllocatorPool {
            internal: std::ptr::null_mut(),
        }
    }
}

/// Represents single memory allocation.
///
/// It may be either dedicated block of `erupt::vk::DeviceMemory` or a specific region of a
/// bigger block of this type plus unique offset.
///
/// Although the library provides convenience functions that create a Vulkan buffer or image,
/// allocate memory for it and bind them together, binding of the allocation to a buffer or an
/// image is out of scope of the allocation itself.
///
/// Allocation object can exist without buffer/image bound, binding can be done manually by
/// the user, and destruction of it can be done independently of destruction of the allocation.
///
/// The object also remembers its size and some other information. To retrieve this information,
/// use `Allocator::get_allocation_info`.
///
/// Some kinds allocations can be in lost state.
#[derive(Debug, Copy, Clone)]
pub struct Allocation {
    /// Pointer to internal VmaAllocation instance
    internal: ffi::VmaAllocation,
}

impl Allocation {
    pub fn null() -> Allocation {
        Allocation {
            internal: std::ptr::null_mut(),
        }
    }
}

impl Default for Allocation {
    fn default() -> Self {
        Self::null()
    }
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

/// Parameters of `Allocation` objects, that can be retrieved using `Allocator::get_allocation_info`.
#[derive(Default, Debug, Clone)]
pub struct AllocationInfo {
    /// Pointer to internal VmaAllocationInfo instance
    internal: ffi::VmaAllocationInfo,
}

unsafe impl Send for AllocationInfo {}
unsafe impl Sync for AllocationInfo {}

impl AllocationInfo {
    #[inline(always)]
    // Gets the memory type index that this allocation was allocated from. (Never changes)
    pub fn get_memory_type(&self) -> u32 {
        self.internal.memoryType
    }

    /// Handle to Vulkan memory object.
    ///
    /// Same memory object can be shared by multiple allocations.
    ///
    /// It can change after call to `Allocator::defragment` if this allocation is passed
    /// to the function, or if allocation is lost.
    ///
    /// If the allocation is lost, it is equal to `erupt::vk::DeviceMemory::null()`.
    #[inline(always)]
    pub fn get_device_memory(&self) -> erupt::vk::DeviceMemory {
        erupt::vk::DeviceMemory(self.internal.deviceMemory as u64)
    }

    /// Offset into device memory object to the beginning of this allocation, in bytes.
    /// (`self.get_device_memory()`, `self.get_offset()`) pair is unique to this allocation.
    ///
    /// It can change after call to `Allocator::defragment` if this allocation is passed
    /// to the function, or if allocation is lost.
    #[inline(always)]
    pub fn get_offset(&self) -> erupt::vk::DeviceSize {
        self.internal.offset as erupt::vk::DeviceSize
    }

    /// Size of this allocation, in bytes.
    ///
    /// It never changes, unless allocation is lost.
    #[inline(always)]
    pub fn get_size(&self) -> erupt::vk::DeviceSize {
        self.internal.size as erupt::vk::DeviceSize
    }

    /// Pointer to the beginning of this allocation as mapped data.
    ///
    /// If the allocation hasn't been mapped using `Allocator::map_memory` and hasn't been
    /// created with `AllocationCreateFlags::MAPPED` flag, this value is null.
    ///
    /// It can change after call to `Allocator::map_memory`, `Allocator::unmap_memory`.
    /// It can also change after call to `Allocator::defragment` if this allocation is
    /// passed to the function.
    #[inline(always)]
    pub fn get_mapped_data(&self) -> *mut u8 {
        self.internal.pMappedData as *mut u8
    }

    /*#[inline(always)]
    pub fn get_mapped_slice(&self) -> Option<&mut &[u8]> {
        if self.internal.pMappedData.is_null() {
            None
        } else {
            Some(unsafe { &mut ::std::slice::from_raw_parts(self.internal.pMappedData as *mut u8, self.get_size()) })
        }
    }*/

    /// Custom general-purpose pointer that was passed as `AllocationCreateInfo::user_data` or set using `Allocator::set_allocation_user_data`.
    ///
    /// It can change after a call to `Allocator::set_allocation_user_data` for this allocation.
    #[inline(always)]
    pub fn get_user_data(&self) -> *mut ::std::os::raw::c_void {
        self.internal.pUserData
    }

    #[inline(always)]
    pub fn get_name(&self) -> &std::ffi::CStr {
        if self.internal.pName.is_null() {
            Default::default()
        } else {
            unsafe { std::ffi::CStr::from_ptr(self.internal.pName) }
        }
    }
}

bitflags! {
    /// Flags for configuring `Allocator` construction.
    pub struct AllocatorCreateFlags: u32 {

        /// Defaults
        const NONE = 0;

        /// Enables usage of VK_KHR_dedicated_allocation extension.
        ///
        /// The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
        /// When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.
        ///
        /// Using this extension will automatically allocate dedicated blocks of memory for
        /// some buffers and images instead of suballocating place for them out of bigger
        /// memory blocks (as if you explicitly used #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
        /// flag) when it is recommended by the driver. It may improve performance on some
        /// GPUs.
        ///
        /// You may set this flag only if you found out that following device extensions are
        /// supported, you enabled them while creating Vulkan device passed as
        /// VmaAllocatorCreateInfo::device, and you want them to be used internally by this
        /// library:
        ///
        /// - VK_KHR_get_memory_requirements2 (device extension)
        /// - VK_KHR_dedicated_allocation (device extension)
        ///
        /// When this flag is set, you can experience following warnings reported by Vulkan
        /// validation layer. You can ignore them.
        ///
        /// > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
        const KHR_DEDICATED_ALLOCATION = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
        /// Enables usage of VK_KHR_bind_memory2 extension.
        ///
        /// The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
        /// When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.
        ///
        /// You may set this flag only if you found out that this device extension is supported,
        /// you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
        /// and you want it to be used internally by this library.
        ///
        /// The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
        /// which allow to pass a chain of `pNext` structures while binding.
        /// This flag is required if you use `pNext` parameter in vmaBindBufferMemory2() or vmaBindImageMemory2().
        const KHR_BIND_MEMORY2 = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
        /// Enables usage of VK_EXT_memory_budget extension.
        ///
        /// You may set this flag only if you found out that this device extension is supported,
        /// you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
        /// and you want it to be used internally by this library, along with another instance extension
        /// VK_KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).
        ///
        /// The extension provides query for current memory usage and budget, which will probably
        /// be more accurate than an estimation used by the library otherwise.
        const EXT_MEMORY_BUDGET = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        /// Enables usage of VK_AMD_device_coherent_memory extension.
        ///
        /// You may set this flag only if you:
        ///
        /// - found out that this device extension is supported and enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
        /// - checked that `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true and set it while creating the Vulkan device,
        /// - want it to be used internally by this library.
        ///
        /// The extension and accompanying device feature provide access to memory types with
        /// `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flags.
        /// They are useful mostly for writing breadcrumb markers - a common method for debugging GPU crash/hang/TDR.
        ///
        /// When the extension is not enabled, such memory types are still enumerated, but their usage is illegal.
        /// To protect from this error, if you don't create the allocator with this flag, it will refuse to allocate any memory or create a custom pool in such memory type,
        /// returning `VK_ERROR_FEATURE_NOT_PRESENT`.
        const AMD_DEVICE_COHERENT_MEMORY = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
        /// Enables usage of "buffer device address" feature, which allows you to use function
        /// `vkGetBufferDeviceAddress*` to get raw GPU pointer to a buffer and pass it for usage inside a shader.
        ///
        /// You may set this flag only if you:
        ///
        /// 1. (For Vulkan version < 1.2) Found as available and enabled device extension
        /// VK_KHR_buffer_device_address.
        /// This extension is promoted to core Vulkan 1.2.
        /// 2. Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`.
        ///
        /// When this flag is set, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` using VMA.
        /// The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` to
        /// allocated memory blocks wherever it might be needed.
        ///
        /// For more information, see documentation chapter \ref enabling_buffer_device_address.
        const BUFFER_DEVICE_ADDRESS = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        /// Enables usage of VK_EXT_memory_priority extension in the library.
        ///
        /// You may set this flag only if you found available and enabled this device extension,
        /// along with `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority == VK_TRUE`,
        /// while creating Vulkan device passed as VmaAllocatorCreateInfo::device.
        ///
        /// When this flag is used, VmaAllocationCreateInfo::priority and VmaPoolCreateInfo::priority
        /// are used to set priorities of allocated Vulkan memory. Without it, these variables are ignored.
        ///
        /// A priority must be a floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
        /// Larger values are higher priority. The granularity of the priorities is implementation-dependent.
        /// It is automatically passed to every call to `vkAllocateMemory` done by the library using structure `VkMemoryPriorityAllocateInfoEXT`.
        /// The value to be used for default priority is 0.5.
        /// For more details, see the documentation of the VK_EXT_memory_priority extension.
        const EXT_MEMORY_PRIORITY = ffi::VmaAllocatorCreateFlagBits_VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
}

/// Construct `AllocatorCreateFlags` with default values
impl Default for AllocatorCreateFlags {
    fn default() -> Self {
        AllocatorCreateFlags::NONE
    }
}

/// Description of an `Allocator` to be created.
pub struct AllocatorCreateInfo {
    /// Vulkan physical device. It must be valid throughout whole lifetime of created allocator.
    pub physical_device: erupt::vk::PhysicalDevice,

    /// Vulkan device. It must be valid throughout whole lifetime of created allocator.
    pub device: Arc<erupt::DeviceLoader>,

    /// Vulkan instance. It must be valid throughout whole lifetime of created allocator.
    pub instance: Arc<erupt::InstanceLoader>,

    /// Flags for created allocator.
    pub flags: AllocatorCreateFlags,

    /// Preferred size of a single `erupt::vk::DeviceMemory` block to be allocated from large heaps > 1 GiB.
    /// Set to 0 to use default, which is currently 256 MiB.
    pub preferred_large_heap_block_size: usize,

    /// Either empty or an array of limits on maximum number of bytes that can be allocated
    /// out of particular Vulkan memory heap.
    ///
    /// If not empty, it must contain `erupt::vk::PhysicalDeviceMemoryProperties::memory_heap_count` elements,
    /// defining limit on maximum number of bytes that can be allocated out of particular Vulkan
    /// memory heap.
    ///
    /// Any of the elements may be equal to `erupt::vk::WHOLE_SIZE`, which means no limit on that
    /// heap. This is also the default in case of an empty slice.
    ///
    /// If there is a limit defined for a heap:
    ///
    /// * If user tries to allocate more memory from that heap using this allocator, the allocation
    /// fails with `erupt::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY`.
    ///
    /// * If the limit is smaller than heap size reported in `erupt::vk::MemoryHeap::size`, the value of this
    /// limit will be reported instead when using `Allocator::get_memory_properties`.
    ///
    /// Warning! Using this feature may not be equivalent to installing a GPU with smaller amount of
    /// memory, because graphics driver doesn't necessary fail new allocations with
    /// `erupt::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is exceeded. It may return success
    /// and just silently migrate some device memory" blocks to system RAM. This driver behavior can
    /// also be controlled using the `VK_AMD_memory_overallocation_behavior` extension.
    pub heap_size_limits: Option<Vec<erupt::vk::DeviceSize>>,

    /// Optional. The highest version of Vulkan that the application is designed to use.
    ///
    /// It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    /// The patch version number specified is ignored. Only the major and minor versions are considered.
    /// It must be less or equal (preferably equal) to value as passed to `vkCreateInstance` as `VkApplicationInfo::apiVersion`.
    /// Only versions 1.0, 1.1, 1.2, 1.3 are supported by the current implementation.
    /// Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    pub vulkan_api_version: u32,
}

// /// Construct `AllocatorCreateInfo` with default values
// ///
// /// Note that the default `device` and `instance` fields are filled with dummy
// /// implementations that will panic if used. These fields must be overwritten.
// impl Default for AllocatorCreateInfo {
//     fn default() -> Self {
//         extern "C" fn get_device_proc_addr(
//             _: erupt::vk::Instance,
//             _: *const std::os::raw::c_char,
//         ) -> *const std::os::raw::c_void {
//             std::ptr::null()
//         }
//         extern "C" fn get_instance_proc_addr(
//             _: erupt::vk::Instance,
//             _: *const std::os::raw::c_char,
//         ) -> *const std::os::raw::c_void {
//             get_device_proc_addr as *const _
//         }
//         let instance = Arc<>;
//         let device = unsafe { Arc::new(DeviceLoader::) };
//         AllocatorCreateInfo {
//             physical_device: erupt::vk::PhysicalDevice::null(),
//             device,
//             instance,
//             flags: AllocatorCreateFlags::NONE,
//             preferred_large_heap_block_size: 0,
//             frame_in_use_count: 0,
//             heap_size_limits: None,
//         }
//     }
// }

/// Converts a raw result into an erupt result.
#[inline]
fn ffi_to_result(result: ffi::VkResult) -> erupt::vk::Result {
    erupt::vk::Result(result)
}

/// Converts an `AllocationCreateInfo` struct into the raw representation.
fn allocation_create_info_to_ffi(info: &AllocationCreateInfo) -> ffi::VmaAllocationCreateInfo {
    ffi::VmaAllocationCreateInfo {
        usage: info.usage as u32,
        flags: info.flags.bits(),
        requiredFlags: info.required_flags.bits(),
        preferredFlags: info.preferred_flags.bits(),
        memoryTypeBits: info.memory_type_bits,
        pool: match &info.pool {
            Some(pool) => pool.internal,
            None => std::ptr::null_mut(),
        },
        pUserData: info.user_data.unwrap_or(::std::ptr::null_mut()),
        priority: info.priority,
    }
}

/// Converts an `AllocatorPoolCreateInfo` struct into the raw representation.
fn pool_create_info_to_ffi(info: &AllocatorPoolCreateInfo) -> ffi::VmaPoolCreateInfo {
    ffi::VmaPoolCreateInfo {
        memoryTypeIndex: info.memory_type_index,
        flags: info.flags.bits(),
        blockSize: info.block_size as ffi::VkDeviceSize,
        minBlockCount: info.min_block_count,
        maxBlockCount: info.max_block_count,
        minAllocationAlignment: info.min_allocation_alignment,
        priority: info.priority,
        pMemoryAllocateNext: std::ptr::null_mut(),
    }
}

/// Intended usage of memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum MemoryUsage {
    /// No intended memory usage specified.
    /// Use other members of `AllocationCreateInfo` to specify your requirements.
    Unknown = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_UNKNOWN,

    /// Memory will be used on device only, so fast access from the device is preferred.
    /// It usually means device-local GPU (video) memory.
    /// No need to be mappable on host.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_DEFAULT`.
    ///
    /// Usage:
    ///
    /// - Resources written and read by device, e.g. images used as attachments.
    /// - Resources transferred from host once (immutable) or infrequently and read by
    ///   device multiple times, e.g. textures to be sampled, vertex buffers, uniform
    ///   (constant) buffers, and majority of other types of resources used on GPU.
    ///
    /// Allocation may still end up in `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE` memory on some implementations.
    /// In such case, you are free to map it.
    /// You can use `AllocationCreateFlags::MAPPED` with this usage type.
    #[deprecated = "Obsolete, preserved for backward compatibility. Prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`."]
    GpuOnly = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_GPU_ONLY,

    /// Memory will be mappable on host.
    /// It usually means CPU (system) memory.
    /// Guarantees to be `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE` and `erupt::vk::MemoryPropertyFlags::HOST_COHERENT`.
    /// CPU access is typically uncached. Writes may be write-combined.
    /// Resources created in this pool may still be accessible to the device, but access to them can be slow.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_UPLOAD`.
    ///
    /// Usage: Staging copy of resources used as transfer source.
    #[deprecated = "Obsolete, preserved for backward compatibility. Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`."]
    CpuOnly = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_CPU_ONLY,

    /// Memory that is both mappable on host (guarantees to be `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE`) and preferably fast to access by GPU.
    /// CPU access is typically uncached. Writes may be write-combined.
    ///
    /// Usage: Resources written frequently by host (dynamic), read by device. E.g. textures, vertex buffers,
    /// uniform buffers updated every frame or every draw call.
    #[deprecated = "Obsolete, preserved for backward compatibility. Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`."]
    CpuToGpu = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_CPU_TO_GPU,

    /// Memory mappable on host (guarantees to be `erupt::vk::MemoryPropertFlags::HOST_VISIBLE`) and cached.
    /// It is roughly equivalent of `D3D12_HEAP_TYPE_READBACK`.
    ///
    /// Usage:
    ///
    /// - Resources written by device, read by host - results of some computations, e.g. screen capture, average scene luminance for HDR tone mapping.
    /// - Any resources read or accessed randomly on host, e.g. CPU-side copy of vertex buffer used as source of transfer, but also used for collision detection.
    #[deprecated = "Obsolete, preserved for backward compatibility. Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`."]
    GpuToCpu = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_GPU_TO_CPU,

    /// CPU memory - memory that is preferably not `DEVICE_LOCAL`, but also not guaranteed to be `HOST_VISIBLE`.
    ///
    /// Usage: Staging copy of resources moved from GPU memory to CPU memory as part
    /// of custom paging/residency mechanism, to be moved back to GPU memory when needed.
    #[deprecated = "Obsolete, preserved for backward compatibility. Prefers not `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`."]
    CpuCopy = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_CPU_COPY,

    /// Lazily allocated GPU memory having `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    /// Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.
    ///
    /// Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.
    ///
    /// Allocations with this usage are always created as dedicated - it implies #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    GpuLazilyAllocated = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED,

    /// Selects best memory type automatically.
    /// This flag is recommended for most common use cases.
    ///
    /// When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    /// you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    /// in VmaAllocationCreateInfo::flags.
    ///
    /// It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    /// vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    /// and not with generic memory allocation functions.
    Auto = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_AUTO,

    /// Selects best memory type automatically with preference for GPU (device) memory.
    ///
    /// When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    /// you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    /// in VmaAllocationCreateInfo::flags.
    ///
    /// It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    /// vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    /// and not with generic memory allocation functions.
    AutoPreferDevice = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,

    /// Selects best memory type automatically with preference for CPU (host) memory.
    ///
    /// When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    /// you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    /// in VmaAllocationCreateInfo::flags.
    ///
    /// It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    /// vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    /// and not with generic memory allocation functions.
    AutoPreferHost = ffi::VmaMemoryUsage_VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self::Unknown
    }
}

bitflags! {
    /// Flags for configuring `AllocatorPool` construction.
    pub struct AllocatorPoolCreateFlags: u32 {
        const NONE = 0;

        /// Use this flag if you always allocate only buffers and linear images or only optimal images
        /// out of this pool and so buffer-image granularity can be ignored.
        ///
        /// This is an optional optimization flag.
        ///
        /// If you always allocate using `Allocator::create_buffer`, `Allocator::create_image`,
        /// `Allocator::allocate_memory_for_buffer`, then you don't need to use it because allocator
        /// knows exact type of your allocations so it can handle buffer-image granularity
        /// in the optimal way.
        ///
        /// If you also allocate using `Allocator::allocate_memory_for_image` or `Allocator::allocate_memory`,
        /// exact type of such allocations is not known, so allocator must be conservative
        /// in handling buffer-image granularity, which can lead to suboptimal allocation
        /// (wasted memory). In that case, if you can make sure you always allocate only
        /// buffers and linear images or only optimal images out of this pool, use this flag
        /// to make allocator disregard buffer-image granularity and so make allocations
        /// faster and more optimal.
        const IGNORE_BUFFER_IMAGE_GRANULARITY = ffi::VmaPoolCreateFlagBits_VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT;

        /// Enables alternative, linear allocation algorithm in this pool.
        ///
        /// Specify this flag to enable linear allocation algorithm, which always creates
        /// new allocations after last one and doesn't reuse space from allocations freed in
        /// between. It trades memory consumption for simplified algorithm and data
        /// structure, which has better performance and uses less memory for metadata.
        ///
        /// By using this flag, you can achieve behavior of free-at-once, stack,
        /// ring buffer, and double stack.
        const LINEAR_ALGORITHM = ffi::VmaPoolCreateFlagBits_VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;


        /// Bit mask to extract only `*_ALGORITHM` bits from entire set of flags.
        const ALGORITHM_MASK = ffi::VmaPoolCreateFlagBits_VMA_POOL_CREATE_ALGORITHM_MASK;
    }
}

impl Default for AllocatorPoolCreateFlags {
    fn default() -> Self {
        Self::NONE
    }
}

bitflags! {
    /// Flags for configuring `Allocation` construction.
    pub struct AllocationCreateFlags: u32 {
        /// Default configuration for allocation.
        const NONE = 0x0000_0000;

        /// Set this flag if the allocation should have its own memory block.
        ///
        /// Use it for special, big resources, like fullscreen images used as attachments.
        ///
        /// You should not use this flag if `AllocationCreateInfo::pool` is not `None`.
        const DEDICATED_MEMORY = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

        /// Set this flag to only try to allocate from existing `erupt::vk::DeviceMemory` blocks and never create new such block.
        ///
        /// If new allocation cannot be placed in any of the existing blocks, allocation
        /// fails with `erupt::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY` error.
        ///
        /// You should not use `AllocationCreateFlags::DEDICATED_MEMORY` and `AllocationCreateFlags::NEVER_ALLOCATE` at the same time. It makes no sense.
        ///
        /// If `AllocationCreateInfo::pool` is not `None`, this flag is implied and ignored.
        const NEVER_ALLOCATE = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;

        /// Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.
        ///
        /// Pointer to mapped memory will be returned through `Allocation::get_mapped_data()`.
        ///
        /// Is it valid to use this flag for allocation made from memory type that is not
        /// `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
        /// useful if you need an allocation that is efficient to use on GPU
        /// (`erupt::vk::MemoryPropertyFlags::DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
        /// support it (e.g. Intel GPU).
        const MAPPED = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_MAPPED_BIT;

        /// Set this flag to treat `AllocationCreateInfo::user_data` as pointer to a
        /// null-terminated string. Instead of copying pointer value, a local copy of the
        /// string is made and stored in allocation's user data. The string is automatically
        /// freed together with the allocation. It is also used in `Allocator::build_stats_string`.
        #[deprecated = "Preserved for backward compatibility. Consider using vmaSetAllocationName() instead"]
        const USER_DATA_COPY_STRING = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;

        /// Allocation will be created from upper stack in a double stack pool.
        ///
        /// This flag is only allowed for custom pools created with `AllocatorPoolCreateFlags::LINEAR_ALGORITHM` flag.
        const UPPER_ADDRESS = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT;

        /// Create both buffer/image and allocation, but don't bind them together.
        /// It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
        /// The flag is meaningful only with functions that bind by default: vmaCreateBuffer(), vmaCreateImage().
        /// Otherwise it is ignored.
        ///
        /// If you want to make sure the new buffer/image is not tied to the new memory allocation
        /// through `VkMemoryDedicatedAllocateInfoKHR` structure in case the allocation ends up in its own memory block,
        /// use also flag #VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT.
        const DONT_BIND = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_DONT_BIND_BIT;

        /// Create allocation only if additional device memory required for it, if any, won't exceed
        /// memory budget. Otherwise return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
        const WITHIN_BUDGET = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

        /// Set this flag if the allocated memory will have aliasing resources.
        ///
        /// Usage of this flag prevents supplying `VkMemoryDedicatedAllocateInfoKHR` when #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is specified.
        /// Otherwise created dedicated memory will not be suitable for aliasing resources, resulting in Vulkan Validation Layer errors.
        const CAN_ALIAS = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;

        /// Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).
        ///
        /// - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
        /// you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
        /// - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
        /// This includes allocations created in \ref custom_memory_pools.
        ///
        /// Declares that mapped memory will only be written sequentially, e.g. using `memcpy()` or a loop writing number-by-number,
        /// never read or accessed randomly, so a memory type can be selected that is uncached and write-combined.
        ///
        /// Warning: Violating this declaration may work correctly, but will likely be very slow.
        /// Watch out for implicit reads introduced by doing e.g. `pMappedData[i] += x;`
        /// Better prepare your data in a local variable and `memcpy()` it to the mapped pointer all at once.
        const HOST_ACCESS_SEQUENTIAL_WRITE = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        /// Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).
        ///
        /// - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
        /// you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
        /// - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
        /// This includes allocations created in \ref custom_memory_pools.
        ///
        /// Declares that mapped memory can be read, written, and accessed in random order,
        /// so a `HOST_CACHED` memory type is required.
        const HOST_ACCESS_RANDOM = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

        /// Together with #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        /// it says that despite request for host access, a not-`HOST_VISIBLE` memory type can be selected
        /// if it may improve performance.
        ///
        /// By using this flag, you declare that you will check if the allocation ended up in a `HOST_VISIBLE` memory type
        /// (e.g. using vmaGetAllocationMemoryProperties()) and if not, you will create some "staging" buffer and
        /// issue an explicit transfer to write/read your data.
        /// To prepare for this possibility, don't forget to add appropriate flags like
        /// `VK_BUFFER_USAGE_TRANSFER_DST_BIT`, `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to the parameters of created buffer or image.
        const HOST_ACCESS_ALLOW_TRANSFER_INSTEAD = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT;

        /// Allocation strategy that chooses always the lowest offset in available space.
        /// This is not the most efficient strategy but achieves highly packed data.
        /// Used internally by defragmentation, not recomended in typical usage.
        const STRATEGY_MIN_OFFSET = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT;

        /// Allocation strategy that chooses smallest possible free range for the
        /// allocation.
        const STRATEGY_BEST_FIT = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT;

        /// Allocation strategy that chooses first suitable free range for the
        /// allocation.
        ///
        /// "First" doesn't necessarily means the one with smallest offset in memory,
        /// but rather the one that is easiest and fastest to find.
        const STRATEGY_FIRST_FIT = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT;

        /// Allocation strategy that tries to minimize memory usage.
        const STRATEGY_MIN_MEMORY = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;

        /// Allocation strategy that tries to minimize allocation time.
        const STRATEGY_MIN_TIME = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;

        /// A bit mask to extract only `*_STRATEGY` bits from entire set of flags.
        const STRATEGY_MASK = ffi::VmaAllocationCreateFlagBits_VMA_ALLOCATION_CREATE_STRATEGY_MASK;
    }
}

impl Default for AllocationCreateFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// Description of an `Allocation` to be created.
#[derive(Default, Debug, Clone)]
pub struct AllocationCreateInfo {
    /// Flags for configuring the allocation
    pub flags: AllocationCreateFlags,

    /// Intended usage of memory.
    ///
    /// You can leave `MemoryUsage::UNKNOWN` if you specify memory requirements
    /// in another way.
    ///
    /// If `pool` is not `None`, this member is ignored.
    pub usage: MemoryUsage,

    /// Flags that must be set in a Memory Type chosen for an allocation.
    ///
    /// Leave 0 if you specify memory requirements in other way.
    ///
    /// If `pool` is not `None`, this member is ignored.
    pub required_flags: erupt::vk::MemoryPropertyFlags,

    /// Flags that preferably should be set in a memory type chosen for an allocation.
    ///
    /// Set to 0 if no additional flags are prefered.
    ///
    /// If `pool` is not `None`, this member is ignored.
    pub preferred_flags: erupt::vk::MemoryPropertyFlags,

    /// Bit mask containing one bit set for every memory type acceptable for this allocation.
    ///
    /// Value 0 is equivalent to `std::u32::MAX` - it means any memory type is accepted if
    /// it meets other requirements specified by this structure, with no further restrictions
    /// on memory type index.
    ///
    /// If `pool` is not `None`, this member is ignored.
    pub memory_type_bits: u32,

    /// Pool that this allocation should be created in.
    ///
    /// Specify `None` to allocate from default pool. If not `None`, members:
    /// `usage`, `required_flags`, `preferred_flags`, `memory_type_bits` are ignored.
    pub pool: Option<AllocatorPool>,

    /// Custom general-purpose pointer that will be stored in `Allocation`, can be read
    /// as `Allocation::get_user_data()` and changed using `Allocator::set_allocation_user_data`.
    ///
    /// If `AllocationCreateFlags::USER_DATA_COPY_STRING` is used, it must be either null or pointer to a
    /// null-terminated string. The string will be then copied to internal buffer, so it
    /// doesn't need to be valid after allocation call.
    pub user_data: Option<*mut ::std::os::raw::c_void>,

    /// A floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
    ///
    /// It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object
    /// and this allocation ends up as dedicated or is explicitly forced as dedicated using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    /// Otherwise, it has the priority of a memory block where it is placed and this variable is ignored.
    pub priority: f32,
}

/// Description of an `AllocationPool` to be created.
#[derive(Default, Debug, Clone)]
pub struct AllocatorPoolCreateInfo {
    /// Vulkan memory type index to allocate this pool from.
    pub memory_type_index: u32,

    /// Use combination of `AllocatorPoolCreateFlags`
    pub flags: AllocatorPoolCreateFlags,

    /// Size of a single `erupt::vk::DeviceMemory` block to be allocated as part of this
    /// pool, in bytes.
    ///
    /// Specify non-zero to set explicit, constant size of memory blocks used by
    /// this pool.
    ///
    /// Leave 0 to use default and let the library manage block sizes automatically.
    /// Sizes of particular blocks may vary.
    pub block_size: usize,

    /// Minimum number of blocks to be always allocated in this pool, even if they stay empty.
    ///
    /// Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    pub min_block_count: usize,

    /// Maximum number of blocks that can be allocated in this pool.
    ///
    /// Set to 0 to use default, which is no limit.
    ///
    /// Set to same value as `AllocatorPoolCreateInfo::min_block_count` to have fixed amount
    /// of memory allocated throughout whole lifetime of this pool.
    pub max_block_count: usize,

    /// A floating-point value between 0 and 1, indicating the priority of the allocations in this pool relative to other memory allocations.
    ///
    /// It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object.
    /// Otherwise, this variable is ignored.
    pub priority: f32,

    /// \brief Additional minimum alignment to be used for all allocations created from this pool. Can be 0.
    ///
    /// Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.
    /// It can be useful in cases where alignment returned by Vulkan by functions like `vkGetBufferMemoryRequirements` is not enough,
    /// e.g. when doing interop with OpenGL.
    pub min_allocation_alignment: erupt::vk::DeviceSize,
}

bitflags! {
    /// Flags for configuring the defragmentation process.
    pub struct DefragmentationFlags: u32 {
        /// Default configuration for allocation.
        const NONE = 0x0000_0000;

        /// Use simple but fast algorithm for defragmentation.
        /// May not achieve best results but will require least time to compute and least allocations to copy.
        const ALGORITHM_FAST = ffi::VmaDefragmentationFlagBits_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;
        /// Default defragmentation algorithm, applied also when no `ALGORITHM` flag is specified.
        /// Offers a balance between defragmentation quality and the amount of allocations and bytes that need to be moved.
        const ALGORITHM_BALANCED = ffi::VmaDefragmentationFlagBits_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
        /// Perform full defragmentation of memory.
        /// Can result in notably more time to compute and allocations to copy, but will achieve best memory packing.
        const ALGORITHM_FULL = ffi::VmaDefragmentationFlagBits_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT;
        /// Use the most roboust algorithm at the cost of time to compute and number of copies to make.
        /// Only available when bufferImageGranularity is greater than 1, since it aims to reduce
        /// alignment issues between different types of resources.
        /// Otherwise falls back to same behavior as #VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT.
        const ALGORITHM_EXTENSIVE = ffi::VmaDefragmentationFlagBits_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT;

        /// A bit mask to extract only `ALGORITHM` bits from entire set of flags.
        const ALGORITHM_MASK = ffi::VmaDefragmentationFlagBits_VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK;
    }
}

impl Default for DefragmentationFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// Operation performed on single defragmentation move.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefragmentationMoveOperation {
    /// Buffer/image has been recreated at `dstTmpAllocation`, data has been copied, old buffer/image has been destroyed. `srcAllocation` should be changed to point to the new place. This is the default value set by vmaBeginDefragmentationPass().
    Copy = ffi::VmaDefragmentationMoveOperation_VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY,
    /// Set this value if you cannot move the allocation. New place reserved at `dstTmpAllocation` will be freed. `srcAllocation` will remain unchanged.
    Ignore = ffi::VmaDefragmentationMoveOperation_VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE,
    /// Set this value if you decide to abandon the allocation and you destroyed the buffer/image. New place reserved at `dstTmpAllocation` will be freed, along with `srcAllocation`, which will be destroyed.
    Destroy = ffi::VmaDefragmentationMoveOperation_VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY,
}

impl TryFrom<u32> for DefragmentationMoveOperation {
    type Error = u32;

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        if value == Self::Copy as u32 {
            Ok(Self::Copy)
        } else if value == Self::Ignore as u32 {
            Ok(Self::Ignore)
        } else if value == Self::Destroy as u32 {
            Ok(Self::Destroy)
        } else {
            Err(value)
        }
    }
}

impl Default for DefragmentationMoveOperation {
    fn default() -> Self {
        Self::Ignore
    }
}

#[derive(Debug)]
pub struct DefragmentationContext {
    internal: ffi::VmaDefragmentationContext,
}

/// Parameters for defragmentation.
///
/// To be used with function `Allocator::defragmentation_begin`.
#[derive(Debug, Clone)]
pub struct DefragmentationInfo {
    /// See `DefragmentationFlags`
    pub flags: DefragmentationFlags,

    /// Custom pool to be defragmented.
    ///
    /// If `None` then default pools will undergo defragmentation process.
    pub pool: Option<AllocatorPool>,

    /// Maximum numbers of bytes that can be copied during single pass, while moving allocations to different places.
    ///
    /// `0` means no limit.
    pub max_bytes_per_pass: erupt::vk::DeviceSize,

    /// Maximum number of allocations that can be moved during single pass to a different place.
    ///
    /// `0` means no limit.
    pub max_allocations_per_pass: u32,
}

/// Single move of an allocation to be done for defragmentation.
#[derive(Debug, Clone)]
pub struct DefragmentationMove {
    /// Operation to be performed on the allocation by vmaEndDefragmentationPass(). Default value is #VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY. You can modify it.
    pub operation: DefragmentationMoveOperation,
    /// Allocation that should be moved.
    pub src_allocation: Allocation,
    /// Temporary allocation pointing to destination memory that will replace `srcAllocation`.
    ///
    /// Warning: Do not store this allocation in your data structures! It exists only temporarily, for the duration of the defragmentation pass,
    /// to be used for binding new buffer/image to the destination memory using e.g. vmaBindBufferMemory().
    /// vmaEndDefragmentationPass() will destroy it and make `srcAllocation` point to this memory.
    pub dst_tmp_allocation: Allocation,
}

pub enum DefragmentationPassResult {
    /// No more moves are possible. You can omit call to vmaEndDefragmentationPass() and simply end whole defragmentation.
    Success,
    /// There are pending moves returned. You need to perform them, call vmaEndDefragmentationPass(), and then preferably try another pass with vmaBeginDefragmentationPass().
    Incomplete(DefragmentationPassMoveInfo),
}

/// Parameters for incremental defragmentation steps.
///
/// To be used with function `Allocator::defragmentation_begin_pass`.
#[derive(Debug, Clone)]
pub struct DefragmentationPassMoveInfo {
    internal: ffi::VmaDefragmentationPassMoveInfo,
    moves: Box<[DefragmentationMove]>,
}

// Slice accessors that prevent changing the array length
impl DefragmentationPassMoveInfo {
    pub fn moves(&self) -> &[DefragmentationMove] {
        &self.moves
    }
    pub fn moves_mut(&mut self) -> &mut [DefragmentationMove] {
        &mut self.moves
    }
}

/// Statistics returned by `Allocator::defragment`
#[derive(Debug, Copy, Clone)]
pub struct DefragmentationStats {
    /// Total number of bytes that have been copied while moving allocations to different places.
    pub bytes_moved: usize,

    /// Total number of bytes that have been released to the system by freeing empty `erupt::vk::DeviceMemory` objects.
    pub bytes_freed: usize,

    /// Number of allocations that have been moved to different places.
    pub allocations_moved: u32,

    /// Number of empty `erupt::vk::DeviceMemory` objects that have been released to the system.
    pub device_memory_blocks_freed: u32,
}

impl Allocator {
    /// Constructor a new `Allocator` using the provided options.
    pub fn new(create_info: &AllocatorCreateInfo) -> Result<Self> {
        let instance = create_info.instance.clone();
        let device = create_info.device.clone();
        let routed_functions = unsafe {
            ffi::VmaVulkanFunctions {
                vkGetInstanceProcAddr: mem::transmute::<_, ffi::PFN_vkGetInstanceProcAddr>(
                    instance.get_instance_proc_addr,
                ),
                vkGetDeviceProcAddr: mem::transmute::<_, ffi::PFN_vkGetDeviceProcAddr>(
                    instance.get_device_proc_addr,
                ),
                vkGetPhysicalDeviceProperties: mem::transmute::<
                    _,
                    ffi::PFN_vkGetPhysicalDeviceProperties,
                >(
                    instance.get_physical_device_properties
                ),
                vkGetPhysicalDeviceMemoryProperties: mem::transmute::<
                    _,
                    ffi::PFN_vkGetPhysicalDeviceMemoryProperties,
                >(
                    instance.get_physical_device_memory_properties,
                ),
                vkAllocateMemory: mem::transmute::<_, ffi::PFN_vkAllocateMemory>(
                    device.allocate_memory,
                ),
                vkFreeMemory: mem::transmute::<_, ffi::PFN_vkFreeMemory>(device.free_memory),
                vkMapMemory: mem::transmute::<_, ffi::PFN_vkMapMemory>(device.map_memory),
                vkUnmapMemory: mem::transmute::<_, ffi::PFN_vkUnmapMemory>(device.unmap_memory),
                vkFlushMappedMemoryRanges: mem::transmute::<_, ffi::PFN_vkFlushMappedMemoryRanges>(
                    device.flush_mapped_memory_ranges,
                ),
                vkInvalidateMappedMemoryRanges: mem::transmute::<
                    _,
                    ffi::PFN_vkInvalidateMappedMemoryRanges,
                >(
                    device.invalidate_mapped_memory_ranges
                ),
                vkBindBufferMemory: mem::transmute::<_, ffi::PFN_vkBindBufferMemory>(
                    device.bind_buffer_memory,
                ),
                vkBindBufferMemory2KHR: mem::transmute::<_, ffi::PFN_vkBindBufferMemory2KHR>(
                    device.bind_buffer_memory2,
                ),
                vkBindImageMemory: mem::transmute::<_, ffi::PFN_vkBindImageMemory>(
                    device.bind_image_memory,
                ),
                vkBindImageMemory2KHR: mem::transmute::<_, ffi::PFN_vkBindImageMemory2KHR>(
                    device.bind_image_memory2,
                ),
                vkGetBufferMemoryRequirements: mem::transmute::<
                    _,
                    ffi::PFN_vkGetBufferMemoryRequirements,
                >(
                    device.get_buffer_memory_requirements
                ),
                vkGetImageMemoryRequirements: mem::transmute::<
                    _,
                    ffi::PFN_vkGetImageMemoryRequirements,
                >(
                    device.get_image_memory_requirements
                ),
                vkCreateBuffer: mem::transmute::<_, ffi::PFN_vkCreateBuffer>(device.create_buffer),
                vkDestroyBuffer: mem::transmute::<_, ffi::PFN_vkDestroyBuffer>(
                    device.destroy_buffer,
                ),
                vkCreateImage: mem::transmute::<_, ffi::PFN_vkCreateImage>(device.create_image),
                vkDestroyImage: mem::transmute::<_, ffi::PFN_vkDestroyImage>(device.destroy_image),
                vkCmdCopyBuffer: mem::transmute::<_, ffi::PFN_vkCmdCopyBuffer>(
                    device.cmd_copy_buffer,
                ),
                vkGetBufferMemoryRequirements2KHR: mem::transmute::<
                    _,
                    ffi::PFN_vkGetBufferMemoryRequirements2KHR,
                >(
                    device.get_buffer_memory_requirements2
                ),
                vkGetImageMemoryRequirements2KHR: mem::transmute::<
                    _,
                    ffi::PFN_vkGetImageMemoryRequirements2KHR,
                >(
                    device.get_image_memory_requirements2
                ),
                vkGetPhysicalDeviceMemoryProperties2KHR: mem::transmute::<
                    _,
                    ffi::PFN_vkGetPhysicalDeviceMemoryProperties2KHR,
                >(
                    instance.get_physical_device_properties2
                ),
                vkGetDeviceBufferMemoryRequirements: mem::transmute::<
                    _,
                    ffi::PFN_vkGetDeviceBufferMemoryRequirements,
                >(
                    device.get_device_buffer_memory_requirements
                ),
                vkGetDeviceImageMemoryRequirements: mem::transmute::<
                    _,
                    ffi::PFN_vkGetDeviceImageMemoryRequirements,
                >(
                    device.get_device_image_memory_requirements
                ),
            }
        };
        let ffi_create_info = ffi::VmaAllocatorCreateInfo {
            physicalDevice: create_info.physical_device.object_handle() as ffi::VkPhysicalDevice,
            device: create_info.device.handle.object_handle() as ffi::VkDevice,
            instance: instance.handle.object_handle() as ffi::VkInstance,
            flags: create_info.flags.bits(),
            preferredLargeHeapBlockSize: create_info.preferred_large_heap_block_size as u64,
            pHeapSizeLimit: match &create_info.heap_size_limits {
                None => ::std::ptr::null(),
                Some(limits) => limits.as_ptr(),
            },
            pVulkanFunctions: &routed_functions,
            pAllocationCallbacks: ::std::ptr::null(), // TODO: Add support
            pDeviceMemoryCallbacks: ::std::ptr::null(), // TODO: Add support
            vulkanApiVersion: create_info.vulkan_api_version,
            pTypeExternalMemoryHandleTypes: ::std::ptr::null(), // TODO: Make configurable
        };
        let mut internal: ffi::VmaAllocator = std::ptr::null_mut();
        let result = ffi_to_result(unsafe {
            ffi::vmaCreateAllocator(
                &ffi_create_info as *const ffi::VmaAllocatorCreateInfo,
                &mut internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(Allocator {
                internal,
                instance,
                device,
            }),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// The allocator fetches `erupt::vk::PhysicalDeviceProperties` from the physical device.
    /// You can get it here, without fetching it again on your own.
    pub fn get_physical_device_properties(&self) -> Result<erupt::vk::PhysicalDeviceProperties> {
        let mut ffi_properties: *const ffi::VkPhysicalDeviceProperties = std::ptr::null();
        Ok(unsafe {
            ffi::vmaGetPhysicalDeviceProperties(self.internal, &mut ffi_properties);
            mem::transmute::<ffi::VkPhysicalDeviceProperties, erupt::vk::PhysicalDeviceProperties>(
                *ffi_properties,
            )
        })
    }

    /// The allocator fetches `erupt::vk::PhysicalDeviceMemoryProperties` from the physical device.
    /// You can get it here, without fetching it again on your own.
    pub fn get_memory_properties(&self) -> Result<erupt::vk::PhysicalDeviceMemoryProperties> {
        let mut ffi_properties: *const ffi::VkPhysicalDeviceMemoryProperties = std::ptr::null();
        Ok(unsafe {
            ffi::vmaGetMemoryProperties(self.internal, &mut ffi_properties);
            mem::transmute::<
                ffi::VkPhysicalDeviceMemoryProperties,
                erupt::vk::PhysicalDeviceMemoryProperties,
            >(*ffi_properties)
        })
    }

    /// Given a memory type index, returns `erupt::vk::MemoryPropertyFlags` of this memory type.
    ///
    /// This is just a convenience function; the same information can be obtained using
    /// `Allocator::get_memory_properties`.
    pub fn get_memory_type_properties(
        &self,
        memory_type_index: u32,
    ) -> Result<erupt::vk::MemoryPropertyFlags> {
        let mut ffi_properties: ffi::VkMemoryPropertyFlags = Default::default();
        Ok(unsafe {
            ffi::vmaGetMemoryTypeProperties(self.internal, memory_type_index, &mut ffi_properties);
            mem::transmute::<ffi::VkMemoryPropertyFlags, erupt::vk::MemoryPropertyFlags>(
                ffi_properties,
            )
        })
    }

    /// Sets index of the current frame.
    ///
    /// This function must be used if you make allocations with `AllocationCreateFlags::CAN_BECOME_LOST` and
    /// `AllocationCreateFlags::CAN_MAKE_OTHER_LOST` flags to inform the allocator when a new frame begins.
    /// Allocations queried using `Allocator::get_allocation_info` cannot become lost
    /// in the current frame.
    pub fn set_current_frame_index(&self, frame_index: u32) {
        unsafe {
            ffi::vmaSetCurrentFrameIndex(self.internal, frame_index);
        }
    }

    /// Retrieves statistics from current state of the `Allocator`.
    pub fn calculate_statistics(&self) -> Result<ffi::VmaTotalStatistics> {
        let mut vma_stats: ffi::VmaTotalStatistics = Default::default();
        unsafe {
            ffi::vmaCalculateStatistics(self.internal, &mut vma_stats as *mut _);
        }
        Ok(vma_stats)
    }

    /// Builds and returns statistics in `JSON` format.
    pub fn build_stats_string(&self, detailed_map: bool) -> Result<String> {
        let mut stats_string: *mut ::std::os::raw::c_char = ::std::ptr::null_mut();
        unsafe {
            ffi::vmaBuildStatsString(
                self.internal,
                &mut stats_string,
                if detailed_map { 1 } else { 0 },
            );
        }
        Ok(if stats_string.is_null() {
            String::new()
        } else {
            let result = unsafe {
                std::ffi::CStr::from_ptr(stats_string)
                    .to_string_lossy()
                    .into_owned()
            };
            unsafe {
                ffi::vmaFreeStatsString(self.internal, stats_string);
            }
            result
        })
    }

    /// Helps to find memory type index, given memory type bits and allocation info.
    ///
    /// This algorithm tries to find a memory type that:
    ///
    /// - Is allowed by memory type bits.
    /// - Contains all the flags from `allocation_info.required_flags`.
    /// - Matches intended usage.
    /// - Has as many flags from `allocation_info.preferred_flags` as possible.
    ///
    /// Returns erupt::vk::Result::ERROR_FEATURE_NOT_PRESENT if not found. Receiving such a result
    /// from this function or any other allocating function probably means that your
    /// device doesn't support any memory type with requested features for the specific
    /// type of resource you want to use it for. Please check parameters of your
    /// resource, like image layout (OPTIMAL versus LINEAR) or mip level count.
    pub fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<u32> {
        let create_info = allocation_create_info_to_ffi(allocation_info);
        let mut memory_type_index: u32 = 0;
        let result = ffi_to_result(unsafe {
            ffi::vmaFindMemoryTypeIndex(
                self.internal,
                memory_type_bits,
                &create_info,
                &mut memory_type_index,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(memory_type_index),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Helps to find memory type index, given buffer info and allocation info.
    ///
    /// It can be useful e.g. to determine value to be used as `AllocatorPoolCreateInfo::memory_type_index`.
    /// It internally creates a temporary, dummy buffer that never has memory bound.
    /// It is just a convenience function, equivalent to calling:
    ///
    /// - `erupt::vk::Device::create_buffer`
    /// - `erupt::vk::Device::get_buffer_memory_requirements`
    /// - `Allocator::find_memory_type_index`
    /// - `erupt::vk::Device::destroy_buffer`
    pub fn find_memory_type_index_for_buffer_info(
        &self,
        buffer_info: &erupt::vk::BufferCreateInfo,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<u32> {
        let allocation_create_info = allocation_create_info_to_ffi(allocation_info);
        let buffer_create_info = unsafe {
            mem::transmute::<erupt::vk::BufferCreateInfo, ffi::VkBufferCreateInfo>(*buffer_info)
        };
        let mut memory_type_index: u32 = 0;
        let result = ffi_to_result(unsafe {
            ffi::vmaFindMemoryTypeIndexForBufferInfo(
                self.internal,
                &buffer_create_info,
                &allocation_create_info,
                &mut memory_type_index,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(memory_type_index),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Helps to find memory type index, given image info and allocation info.
    ///
    /// It can be useful e.g. to determine value to be used as `AllocatorPoolCreateInfo::memory_type_index`.
    /// It internally creates a temporary, dummy image that never has memory bound.
    /// It is just a convenience function, equivalent to calling:
    ///
    /// - `erupt::vk::Device::create_image`
    /// - `erupt::vk::Device::get_image_memory_requirements`
    /// - `Allocator::find_memory_type_index`
    /// - `erupt::vk::Device::destroy_image`
    pub fn find_memory_type_index_for_image_info(
        &self,
        image_info: &erupt::vk::ImageCreateInfo,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<u32> {
        let allocation_create_info = allocation_create_info_to_ffi(allocation_info);
        let image_create_info = unsafe {
            mem::transmute::<erupt::vk::ImageCreateInfo, ffi::VkImageCreateInfo>(*image_info)
        };
        let mut memory_type_index: u32 = 0;
        let result = ffi_to_result(unsafe {
            ffi::vmaFindMemoryTypeIndexForImageInfo(
                self.internal,
                &image_create_info,
                &allocation_create_info,
                &mut memory_type_index,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(memory_type_index),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Allocates Vulkan device memory and creates `AllocatorPool` object.
    pub fn create_pool(&self, pool_info: &AllocatorPoolCreateInfo) -> Result<AllocatorPool> {
        let mut ffi_pool: ffi::VmaPool = std::ptr::null_mut();
        let create_info = pool_create_info_to_ffi(pool_info);
        let result = ffi_to_result(unsafe {
            ffi::vmaCreatePool(self.internal, &create_info, &mut ffi_pool)
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(AllocatorPool { internal: ffi_pool }),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Destroys `AllocatorPool` object and frees Vulkan device memory.
    pub fn destroy_pool(&self, pool: &AllocatorPool) {
        unsafe {
            ffi::vmaDestroyPool(self.internal, pool.internal);
        }
    }

    /// Retrieves statistics of existing `AllocatorPool` object.
    pub fn get_pool_stats(&self, pool: &AllocatorPool) -> Result<ffi::VmaStatistics> {
        let mut pool_stats: ffi::VmaStatistics = Default::default();
        unsafe {
            ffi::vmaGetPoolStatistics(self.internal, pool.internal, &mut pool_stats);
        }
        Ok(pool_stats)
    }

    /// Checks magic number in margins around all allocations in given memory pool in search for corruptions.
    ///
    /// Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
    /// `VMA_DEBUG_MARGIN` is defined to nonzero and the pool is created in memory type that is
    /// `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE` and `erupt::vk::MemoryPropertyFlags::HOST_COHERENT`.
    ///
    /// Possible error values:
    ///
    /// - `erupt::vk::Result::ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for specified pool.
    /// - `erupt::vk::Result::ERROR_VALIDATION_FAILED_EXT` - corruption detection has been performed and found memory corruptions around one of the allocations.
    ///   `VMA_ASSERT` is also fired in that case.
    /// - Other value: Error returned by Vulkan, e.g. memory mapping failure.
    pub fn check_pool_corruption(&self, pool: &AllocatorPool) -> Result<()> {
        let result =
            ffi_to_result(unsafe { ffi::vmaCheckPoolCorruption(self.internal, pool.internal) });
        match result {
            erupt::vk::Result::SUCCESS => Ok(()),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// General purpose memory allocation.
    ///
    /// You should free the memory using `Allocator::free_memory` or 'Allocator::free_memory_pages'.
    ///
    /// It is recommended to use `Allocator::allocate_memory_for_buffer`, `Allocator::allocate_memory_for_image`,
    /// `Allocator::create_buffer`, `Allocator::create_image` instead whenever possible.
    pub fn allocate_memory(
        &self,
        memory_requirements: &erupt::vk::MemoryRequirements,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<(Allocation, AllocationInfo)> {
        let ffi_requirements = unsafe {
            mem::transmute::<erupt::vk::MemoryRequirements, ffi::VkMemoryRequirements>(
                *memory_requirements,
            )
        };
        let create_info = allocation_create_info_to_ffi(allocation_info);
        let mut allocation: Allocation = Default::default();
        let mut allocation_info: AllocationInfo = Default::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaAllocateMemory(
                self.internal,
                &ffi_requirements,
                &create_info,
                &mut allocation.internal,
                &mut allocation_info.internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok((allocation, allocation_info)),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// General purpose memory allocation for multiple allocation objects at once.
    ///
    /// You should free the memory using `Allocator::free_memory` or `Allocator::free_memory_pages`.
    ///
    /// Word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse binding.
    /// It is just a general purpose allocation function able to make multiple allocations at once.
    /// It may be internally optimized to be more efficient than calling `Allocator::allocate_memory` `allocations.len()` times.
    ///
    /// All allocations are made using same parameters. All of them are created out of the same memory pool and type.
    pub fn allocate_memory_pages(
        &self,
        memory_requirements: &erupt::vk::MemoryRequirements,
        allocation_info: &AllocationCreateInfo,
        allocation_count: usize,
    ) -> Result<Vec<(Allocation, AllocationInfo)>> {
        let ffi_requirements = unsafe {
            mem::transmute::<erupt::vk::MemoryRequirements, ffi::VkMemoryRequirements>(
                *memory_requirements,
            )
        };
        let create_info = allocation_create_info_to_ffi(allocation_info);
        let mut allocations: Vec<ffi::VmaAllocation> = vec![std::ptr::null_mut(); allocation_count];
        let mut allocation_info: Vec<ffi::VmaAllocationInfo> =
            vec![Default::default(); allocation_count];
        let result = ffi_to_result(unsafe {
            ffi::vmaAllocateMemoryPages(
                self.internal,
                &ffi_requirements,
                &create_info,
                allocation_count,
                allocations.as_mut_ptr(),
                allocation_info.as_mut_ptr(),
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => {
                let it = allocations.iter().zip(allocation_info.iter());
                let allocations: Vec<(Allocation, AllocationInfo)> = it
                    .map(|(alloc, info)| {
                        (
                            Allocation { internal: *alloc },
                            AllocationInfo { internal: *info },
                        )
                    })
                    .collect();
                Ok(allocations)
            }
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Buffer specialized memory allocation.
    ///
    /// You should free the memory using `Allocator::free_memory` or 'Allocator::free_memory_pages'.
    pub fn allocate_memory_for_buffer(
        &self,
        buffer: erupt::vk::Buffer,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<(Allocation, AllocationInfo)> {
        let ffi_buffer = buffer.object_handle() as ffi::VkBuffer;
        let create_info = allocation_create_info_to_ffi(allocation_info);
        let mut allocation: Allocation = Default::default();
        let mut allocation_info: AllocationInfo = Default::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaAllocateMemoryForBuffer(
                self.internal,
                ffi_buffer,
                &create_info,
                &mut allocation.internal,
                &mut allocation_info.internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok((allocation, allocation_info)),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Image specialized memory allocation.
    ///
    /// You should free the memory using `Allocator::free_memory` or 'Allocator::free_memory_pages'.
    pub fn allocate_memory_for_image(
        &self,
        image: erupt::vk::Image,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<(Allocation, AllocationInfo)> {
        let ffi_image = image.object_handle() as ffi::VkImage;
        let create_info = allocation_create_info_to_ffi(allocation_info);
        let mut allocation: Allocation = Default::default();
        let mut allocation_info: AllocationInfo = Default::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaAllocateMemoryForImage(
                self.internal,
                ffi_image,
                &create_info,
                &mut allocation.internal,
                &mut allocation_info.internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok((allocation, allocation_info)),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Frees memory previously allocated using `Allocator::allocate_memory`,
    /// `Allocator::allocate_memory_for_buffer`, or `Allocator::allocate_memory_for_image`.
    pub fn free_memory(&self, allocation: &Allocation) {
        unsafe {
            ffi::vmaFreeMemory(self.internal, allocation.internal);
        }
    }

    /// Frees memory and destroys multiple allocations.
    ///
    /// Word "pages" is just a suggestion to use this function to free pieces of memory used for sparse binding.
    /// It is just a general purpose function to free memory and destroy allocations made using e.g. `Allocator::allocate_memory',
    /// 'Allocator::allocate_memory_pages` and other functions.
    ///
    /// It may be internally optimized to be more efficient than calling 'Allocator::free_memory` `allocations.len()` times.
    ///
    /// Allocations in 'allocations' slice can come from any memory pools and types.
    pub fn free_memory_pages(&self, allocations: &[Allocation]) {
        let mut allocations_ffi: Vec<ffi::VmaAllocation> =
            allocations.iter().map(|x| x.internal).collect();
        unsafe {
            ffi::vmaFreeMemoryPages(
                self.internal,
                allocations_ffi.len(),
                allocations_ffi.as_mut_ptr(),
            );
        }
    }

    /// Returns current information about specified allocation and atomically marks it as used in current frame.
    ///
    /// Current parameters of given allocation are returned in the result object, available through accessors.
    ///
    /// This function also atomically "touches" allocation - marks it as used in current frame,
    /// just like `Allocator::touch_allocation`.
    ///
    /// If the allocation is in lost state, `allocation.get_device_memory` returns `erupt::vk::DeviceMemory::null()`.
    ///
    /// Although this function uses atomics and doesn't lock any mutex, so it should be quite efficient,
    /// you can avoid calling it too often.
    ///
    /// If you just want to check if allocation is not lost, `Allocator::touch_allocation` will work faster.
    pub fn get_allocation_info(&self, allocation: &Allocation) -> Result<AllocationInfo> {
        let mut allocation_info: AllocationInfo = Default::default();
        unsafe {
            ffi::vmaGetAllocationInfo(
                self.internal,
                allocation.internal,
                &mut allocation_info.internal,
            )
        }
        Ok(allocation_info)
    }

    /// Sets user data in given allocation to new value.
    ///
    /// # Safety
    ///
    /// If the allocation was created with `AllocationCreateFlags::USER_DATA_COPY_STRING`,
    /// `user_data` must be either null, or pointer to a null-terminated string. The function
    /// makes local copy of the string and sets it as allocation's user data. String
    /// passed as user data doesn't need to be valid for whole lifetime of the allocation -
    /// you can free it after this call. String previously pointed by allocation's
    /// user data is freed from memory.
    ///
    /// If the flag was not used, the value of pointer `user_data` is just copied to
    /// allocation's user data. It is opaque, so you can use it however you want - e.g.
    /// as a pointer, ordinal number or some handle to you own data.
    pub unsafe fn set_allocation_user_data(
        &self,
        allocation: &Allocation,
        user_data: *mut ::std::os::raw::c_void,
    ) {
        ffi::vmaSetAllocationUserData(self.internal, allocation.internal, user_data);
    }

    /// Maps memory represented by given allocation and returns pointer to it.
    ///
    /// Maps memory represented by given allocation to make it accessible to CPU code.
    /// When succeeded, result is a pointer to first byte of this memory.
    ///
    /// If the allocation is part of bigger `erupt::vk::DeviceMemory` block, the pointer is
    /// correctly offseted to the beginning of region assigned to this particular
    /// allocation.
    ///
    /// Mapping is internally reference-counted and synchronized, so despite raw Vulkan
    /// function `erupt::vk::Device::MapMemory` cannot be used to map same block of
    /// `erupt::vk::DeviceMemory` multiple times simultaneously, it is safe to call this
    /// function on allocations assigned to the same memory block. Actual Vulkan memory
    /// will be mapped on first mapping and unmapped on last unmapping.
    ///
    /// If the function succeeded, you must call `Allocator::unmap_memory` to unmap the
    /// allocation when mapping is no longer needed or before freeing the allocation, at
    /// the latest.
    ///
    /// It also safe to call this function multiple times on the same allocation. You
    /// must call `Allocator::unmap_memory` same number of times as you called
    /// `Allocator::map_memory`.
    ///
    /// It is also safe to call this function on allocation created with
    /// `AllocationCreateFlags::MAPPED` flag. Its memory stays mapped all the time.
    /// You must still call `Allocator::unmap_memory` same number of times as you called
    /// `Allocator::map_memory`. You must not call `Allocator::unmap_memory` additional
    /// time to free the "0-th" mapping made automatically due to `AllocationCreateFlags::MAPPED` flag.
    ///
    /// This function fails when used on allocation made in memory type that is not
    /// `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE`.
    ///
    /// This function always fails when called for allocation that was created with
    /// `AllocationCreateFlags::CAN_BECOME_LOST` flag. Such allocations cannot be mapped.
    pub fn map_memory(&self, allocation: &Allocation) -> Result<*mut u8> {
        let mut mapped_data: *mut ::std::os::raw::c_void = ::std::ptr::null_mut();
        let result = ffi_to_result(unsafe {
            ffi::vmaMapMemory(self.internal, allocation.internal, &mut mapped_data)
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(mapped_data as *mut u8),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Unmaps memory represented by given allocation, mapped previously using `Allocator::map_memory`.
    pub fn unmap_memory(&self, allocation: &Allocation) {
        unsafe {
            ffi::vmaUnmapMemory(self.internal, allocation.internal);
        }
    }

    /// Flushes memory of given allocation.
    ///
    /// Calls `erupt::vk::Device::FlushMappedMemoryRanges` for memory associated with given range of given allocation.
    ///
    /// - `offset` must be relative to the beginning of allocation.
    /// - `size` can be `erupt::vk::WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
    /// - `offset` and `size` don't have to be aligned; hey are internally rounded down/up to multiple of `nonCoherentAtomSize`.
    /// - If `size` is 0, this call is ignored.
    /// - If memory type that the `allocation` belongs to is not `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE` or it is `erupt::vk::MemoryPropertyFlags::HOST_COHERENT`, this call is ignored.
    pub fn flush_allocation(&self, allocation: &Allocation, offset: usize, size: usize) {
        unsafe {
            ffi::vmaFlushAllocation(
                self.internal,
                allocation.internal,
                offset as ffi::VkDeviceSize,
                size as ffi::VkDeviceSize,
            );
        }
    }

    /// Invalidates memory of given allocation.
    ///
    /// Calls `erupt::vk::Device::invalidate_mapped_memory_ranges` for memory associated with given range of given allocation.
    ///
    /// - `offset` must be relative to the beginning of allocation.
    /// - `size` can be `erupt::vk::WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
    /// - `offset` and `size` don't have to be aligned. They are internally rounded down/up to multiple of `nonCoherentAtomSize`.
    /// - If `size` is 0, this call is ignored.
    /// - If memory type that the `allocation` belongs to is not `erupt::vk::MemoryPropertyFlags::HOST_VISIBLE` or it is `erupt::vk::MemoryPropertyFlags::HOST_COHERENT`, this call is ignored.
    pub fn invalidate_allocation(&self, allocation: &Allocation, offset: usize, size: usize) {
        unsafe {
            ffi::vmaInvalidateAllocation(
                self.internal,
                allocation.internal,
                offset as ffi::VkDeviceSize,
                size as ffi::VkDeviceSize,
            );
        }
    }

    /// Checks magic number in margins around all allocations in given memory types (in both default and custom pools) in search for corruptions.
    ///
    /// `memory_type_bits` bit mask, where each bit set means that a memory type with that index should be checked.
    ///
    /// Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
    /// `VMA_DEBUG_MARGIN` is defined to nonzero and only for memory types that are `HOST_VISIBLE` and `HOST_COHERENT`.
    ///
    /// Possible error values:
    ///
    /// - `erupt::vk::Result::ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for any of specified memory types.
    /// - `erupt::vk::Result::ERROR_VALIDATION_FAILED_EXT` - corruption detection has been performed and found memory corruptions around one of the allocations.
    ///   `VMA_ASSERT` is also fired in that case.
    /// - Other value: Error returned by Vulkan, e.g. memory mapping failure.
    pub fn check_corruption(&self, memory_types: erupt::vk::MemoryPropertyFlags) -> Result<()> {
        let result =
            ffi_to_result(unsafe { ffi::vmaCheckCorruption(self.internal, memory_types.bits()) });
        match result {
            erupt::vk::Result::SUCCESS => Ok(()),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Begins defragmentation process.
    ///
    /// Use this function instead of old, deprecated `Allocator::defragment`.
    ///
    /// Warning! Between the call to `Allocator::defragmentation_begin` and `Allocator::defragmentation_end`.
    ///
    /// - You should not use any of allocations passed as `allocations` or
    /// any allocations that belong to pools passed as `pools`,
    /// including calling `Allocator::get_allocation_info`, `Allocator::touch_allocation`, or access
    /// their data.
    ///
    /// - Some mutexes protecting internal data structures may be locked, so trying to
    /// make or free any allocations, bind buffers or images, map memory, or launch
    /// another simultaneous defragmentation in between may cause stall (when done on
    /// another thread) or deadlock (when done on the same thread), unless you are
    /// 100% sure that defragmented allocations are in different pools.
    ///
    /// - Information returned via stats and `info.allocations_changed` are undefined.
    /// They become valid after call to `Allocator::defragmentation_end`.
    ///
    /// - If `info.command_buffer` is not null, you must submit that command buffer
    /// and make sure it finished execution before calling `Allocator::defragmentation_end`.
    /**
    Interleaved allocations and deallocations of many objects of varying size can
    cause fragmentation over time, which can lead to a situation where the library is unable
    to find a continuous range of free memory for a new allocation despite there is
    enough free space, just scattered across many small free ranges between existing
    allocations.

    To mitigate this problem, you can use defragmentation feature.
    It doesn't happen automatically though and needs your cooperation,
    because VMA is a low level library that only allocates memory.
    It cannot recreate buffers and images in a new place as it doesn't remember the contents of `VkBufferCreateInfo` / `VkImageCreateInfo` structures.
    It cannot copy their contents as it doesn't record any commands to a command buffer.

    Although functions like vmaCreateBuffer(), vmaCreateImage(), vmaDestroyBuffer(), vmaDestroyImage()
    create/destroy an allocation and a buffer/image at once, these are just a shortcut for
    creating the resource, allocating memory, and binding them together.
    Defragmentation works on memory allocations only. You must handle the rest manually.
    Defragmentation is an iterative process that should repreat "passes" as long as related functions
    return `VK_INCOMPLETE` not `VK_SUCCESS`.
    In each pass:

    1. vmaBeginDefragmentationPass() function call:
        - Calculates and returns the list of allocations to be moved in this pass.
            Note this can be a time-consuming process.
        - Reserves destination memory for them by creating temporary destination allocations
            that you can query for their `VkDeviceMemory` + offset using vmaGetAllocationInfo().
    2. Inside the pass, **you should**:
        - Inspect the returned list of allocations to be moved.
        - Create new buffers/images and bind them at the returned destination temporary allocations.
        - Copy data from source to destination resources if necessary.
        - Destroy the source buffers/images, but NOT their allocations.
    3. vmaEndDefragmentationPass() function call:
        - Frees the source memory reserved for the allocations that are moved.
        - Modifies source #VmaAllocation objects that are moved to point to the destination reserved memory.
        - Frees `VkDeviceMemory` blocks that became empty.

    Unlike in previous iterations of the defragmentation API, there is no list of "movable" allocations passed as a parameter.
    Defragmentation algorithm tries to move all suitable allocations.
    You can, however, refuse to move some of them inside a defragmentation pass, by setting
    `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
    This is not recommended and may result in suboptimal packing of the allocations after defragmentation.
    If you cannot ensure any allocation can be moved, it is better to keep movable allocations separate in a custom pool.

    Inside a pass, for each allocation that should be moved:

    - You should copy its data from the source to the destination place by calling e.g. `vkCmdCopyBuffer()`, `vkCmdCopyImage()`.
    - You need to make sure these commands finished executing before destroying the source buffers/images and before calling vmaEndDefragmentationPass().
    - If a resource doesn't contain any meaningful data, e.g. it is a transient color attachment image to be cleared,
    filled, and used temporarily in each rendering frame, you can just recreate this image
    without copying its data.
    - If the resource is in `HOST_VISIBLE` and `HOST_CACHED` memory, you can copy its data on the CPU
    using `memcpy()`.
    - If you cannot move the allocation, you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
    This will cancel the move.
    - vmaEndDefragmentationPass() will then free the destination memory
        not the source memory of the allocation, leaving it unchanged.
    - If you decide the allocation is unimportant and can be destroyed instead of moved (e.g. it wasn't used for long time),
    you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.
    - vmaEndDefragmentationPass() will then free both source and destination memory, and will destroy the source #VmaAllocation object.

    You can defragment a specific custom pool by setting VmaDefragmentationInfo::pool
    (like in the example above) or all the default pools by setting this member to null.

    Defragmentation is always performed in each pool separately.
    Allocations are never moved between different Vulkan memory types.
    The size of the destination memory reserved for a moved allocation is the same as the original one.
    Alignment of an allocation as it was determined using `vkGetBufferMemoryRequirements()` etc. is also respected after defragmentation.
    Buffers/images should be recreated with the same `VkBufferCreateInfo` / `VkImageCreateInfo` parameters as the original ones.

    You can perform the defragmentation incrementally to limit the number of allocations and bytes to be moved
    in each pass, e.g. to call it in sync with render frames and not to experience too big hitches.
    See members: VmaDefragmentationInfo::maxBytesPerPass, VmaDefragmentationInfo::maxAllocationsPerPass.

    It is also safe to perform the defragmentation asynchronously to render frames and other Vulkan and VMA
    usage, possibly from multiple threads, with the exception that allocations
    returned in VmaDefragmentationPassMoveInfo::pMoves shouldn't be destroyed until the defragmentation pass is ended.

    *Mapping* is preserved on allocations that are moved during defragmentation.
    Whether through #VMA_ALLOCATION_CREATE_MAPPED_BIT or vmaMapMemory(), the allocations
    are mapped at their new place. Of course, pointer to the mapped data changes, so it needs to be queried
    using VmaAllocationInfo::pMappedData.

    Note: Defragmentation is not supported in custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT.
     */
    pub fn begin_defragmentation(
        &self,
        info: &DefragmentationInfo,
    ) -> Result<DefragmentationContext> {
        let mut context = DefragmentationContext {
            internal: std::ptr::null_mut(),
        };
        let ffi_info = ffi::VmaDefragmentationInfo {
            flags: info.flags.bits(),
            pool: info
                .pool
                .as_ref()
                .map(|p| p.internal)
                .unwrap_or(std::ptr::null_mut()),
            maxBytesPerPass: info.max_bytes_per_pass,
            maxAllocationsPerPass: info.max_allocations_per_pass,
        };
        let result = ffi_to_result(unsafe {
            ffi::vmaBeginDefragmentation(self.internal, &ffi_info, &mut context.internal)
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(context),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Starts single defragmentation pass.
    pub fn begin_defragmentation_pass(
        &self,
        context: &mut DefragmentationContext,
    ) -> Result<DefragmentationPassResult> {
        let mut ffi_moves = ffi::VmaDefragmentationPassMoveInfo::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaBeginDefragmentationPass(self.internal, context.internal, &mut ffi_moves)
        });
        match result {
            erupt::vk::Result::INCOMPLETE => {
                let mut moves = Vec::with_capacity(ffi_moves.moveCount as usize);
                for i in 0..ffi_moves.moveCount as usize {
                    let ffi_move: ffi::VmaDefragmentationMove =
                        unsafe { std::ptr::read(ffi_moves.pMoves.add(i)) };
                    moves.push(DefragmentationMove {
                        operation: ffi_move.operation.try_into().unwrap(),
                        src_allocation: Allocation {
                            internal: ffi_move.srcAllocation,
                        },
                        dst_tmp_allocation: Allocation {
                            internal: ffi_move.dstTmpAllocation,
                        },
                    })
                }
                Ok(DefragmentationPassResult::Incomplete(
                    DefragmentationPassMoveInfo {
                        internal: ffi_moves,
                        moves: moves.into(),
                    },
                ))
            }
            erupt::vk::Result::SUCCESS => Ok(DefragmentationPassResult::Success),
            error => Err(Error::vulkan(error)),
        }
    }

    /// Ends single defragmentation pass.
    /// Returns true if no more moves are possible, or false if more defragmentations are possible.
    /// Ends incremental defragmentation pass and commits all defragmentation moves from pPassInfo. After this call:
    ///
    ///    - Allocations at moves[i].srcAllocation that had moves[i].operation == Copy (which is the default) will be pointing to the new destination place.
    ///    - Allocation at moves[i].srcAllocation that had moves[i].operation == Destroy will be freed.
    ///
    /// If no more moves are possible you can end whole defragmentation.
    pub fn end_defragmentation_pass(
        &self,
        context: &mut DefragmentationContext,
        moves: &mut DefragmentationPassMoveInfo,
    ) -> Result<bool> {
        for (i, mov) in moves.moves.iter().enumerate() {
            unsafe {
                let ffi_mov = moves.internal.pMoves.add(i);
                std::ptr::addr_of_mut!((*ffi_mov).operation)
                    .write(mov.operation as ffi::VmaDefragmentationMoveOperation);
            }
        }
        let result = ffi_to_result(unsafe {
            ffi::vmaEndDefragmentationPass(self.internal, context.internal, &mut moves.internal)
        });
        if !(result == erupt::vk::Result::SUCCESS || result == erupt::vk::Result::INCOMPLETE) {
            return Err(Error::vulkan(result));
        }
        for (i, mov) in moves.moves.iter_mut().enumerate() {
            unsafe {
                let ffi_mov = moves.internal.pMoves.add(i);
                mov.src_allocation.internal =
                    std::ptr::addr_of_mut!((*ffi_mov).srcAllocation).read();
            }
        }
        Ok(result == erupt::vk::Result::SUCCESS)
    }

    /// Ends defragmentation process.
    ///
    /// Use this function to finish defragmentation started by `Allocator::begin_defragmentation`.
    pub fn end_defragmentation(
        &self,
        context: &mut DefragmentationContext,
    ) -> DefragmentationStats {
        let mut ffi_stats = ffi::VmaDefragmentationStats::default();
        unsafe {
            ffi::vmaEndDefragmentation(self.internal, context.internal, &mut ffi_stats);
        }
        DefragmentationStats {
            bytes_moved: ffi_stats.bytesMoved as usize,
            bytes_freed: ffi_stats.bytesFreed as usize,
            allocations_moved: ffi_stats.allocationsMoved,
            device_memory_blocks_freed: ffi_stats.deviceMemoryBlocksFreed,
        }
    }

    /// Binds buffer to allocation.
    ///
    /// Binds specified buffer to region of memory represented by specified allocation.
    /// Gets `erupt::vk::DeviceMemory` handle and offset from the allocation.
    ///
    /// If you want to create a buffer, allocate memory for it and bind them together separately,
    /// you should use this function for binding instead of `erupt::vk::Device::bind_buffer_memory`,
    /// because it ensures proper synchronization so that when a `erupt::vk::DeviceMemory` object is
    /// used by multiple allocations, calls to `erupt::vk::Device::bind_buffer_memory()` or
    /// `erupt::vk::Device::map_memory()` won't happen from multiple threads simultaneously
    /// (which is illegal in Vulkan).
    ///
    /// It is recommended to use function `Allocator::create_buffer` instead of this one.
    pub fn bind_buffer_memory(
        &self,
        buffer: erupt::vk::Buffer,
        allocation: &Allocation,
    ) -> Result<()> {
        let result = ffi_to_result(unsafe {
            ffi::vmaBindBufferMemory(
                self.internal,
                allocation.internal,
                buffer.object_handle() as ffi::VkBuffer,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(()),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Binds image to allocation.
    ///
    /// Binds specified image to region of memory represented by specified allocation.
    /// Gets `erupt::vk::DeviceMemory` handle and offset from the allocation.
    ///
    /// If you want to create a image, allocate memory for it and bind them together separately,
    /// you should use this function for binding instead of `erupt::vk::Device::bind_image_memory`,
    /// because it ensures proper synchronization so that when a `erupt::vk::DeviceMemory` object is
    /// used by multiple allocations, calls to `erupt::vk::Device::bind_image_memory()` or
    /// `erupt::vk::Device::map_memory()` won't happen from multiple threads simultaneously
    /// (which is illegal in Vulkan).
    ///
    /// It is recommended to use function `Allocator::create_image` instead of this one.
    pub fn bind_image_memory(
        &self,
        image: erupt::vk::Image,
        allocation: &Allocation,
    ) -> Result<()> {
        let result = ffi_to_result(unsafe {
            ffi::vmaBindImageMemory(
                self.internal,
                allocation.internal,
                image.object_handle() as ffi::VkImage,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok(()),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// This function automatically creates a buffer, allocates appropriate memory
    /// for it, and binds the buffer with the memory.
    ///
    /// If the function succeeded, you must destroy both buffer and allocation when you
    /// no longer need them using either convenience function `Allocator::destroy_buffer` or
    /// separately, using `erupt::Device::destroy_buffer` and `Allocator::free_memory`.
    ///
    /// If `AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION` flag was used,
    /// VK_KHR_dedicated_allocation extension is used internally to query driver whether
    /// it requires or prefers the new buffer to have dedicated allocation. If yes,
    /// and if dedicated allocation is possible (AllocationCreateInfo::pool is null
    /// and `AllocationCreateFlags::NEVER_ALLOCATE` is not used), it creates dedicated
    /// allocation for this buffer, just like when using `AllocationCreateFlags::DEDICATED_MEMORY`.
    pub fn create_buffer(
        &self,
        buffer_info: &erupt::vk::BufferCreateInfo,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<(erupt::vk::Buffer, Allocation, AllocationInfo)> {
        let buffer_create_info = unsafe {
            mem::transmute::<erupt::vk::BufferCreateInfo, ffi::VkBufferCreateInfo>(*buffer_info)
        };
        let allocation_create_info = allocation_create_info_to_ffi(allocation_info);
        let mut buffer: ffi::VkBuffer = std::ptr::null_mut();
        let mut allocation: Allocation = Default::default();
        let mut allocation_info: AllocationInfo = Default::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaCreateBuffer(
                self.internal,
                &buffer_create_info,
                &allocation_create_info,
                &mut buffer,
                &mut allocation.internal,
                &mut allocation_info.internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => Ok((
                erupt::vk::Buffer(buffer as u64),
                allocation,
                allocation_info,
            )),
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Destroys Vulkan buffer and frees allocated memory.
    ///
    /// This is just a convenience function equivalent to:
    ///
    /// ```ignore
    /// erupt::vk::Device::destroy_buffer(buffer, None);
    /// Allocator::free_memory(allocator, allocation);
    /// ```
    ///
    /// It it safe to pass null as `buffer` and/or `allocation`.
    pub fn destroy_buffer(&self, buffer: erupt::vk::Buffer, allocation: &Allocation) {
        unsafe {
            ffi::vmaDestroyBuffer(
                self.internal,
                buffer.object_handle() as ffi::VkBuffer,
                allocation.internal,
            );
        }
    }

    /// This function automatically creates an image, allocates appropriate memory
    /// for it, and binds the image with the memory.
    ///
    /// If the function succeeded, you must destroy both image and allocation when you
    /// no longer need them using either convenience function `Allocator::destroy_image` or
    /// separately, using `erupt::Device::destroy_image` and `Allocator::free_memory`.
    ///
    /// If `AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION` flag was used,
    /// `VK_KHR_dedicated_allocation extension` is used internally to query driver whether
    /// it requires or prefers the new image to have dedicated allocation. If yes,
    /// and if dedicated allocation is possible (AllocationCreateInfo::pool is null
    /// and `AllocationCreateFlags::NEVER_ALLOCATE` is not used), it creates dedicated
    /// allocation for this image, just like when using `AllocationCreateFlags::DEDICATED_MEMORY`.
    ///
    /// If `VK_ERROR_VALIDAITON_FAILED_EXT` is returned, VMA may have encountered a problem
    /// that is not caught by the validation layers. One example is if you try to create a 0x0
    /// image, a panic will occur and `VK_ERROR_VALIDAITON_FAILED_EXT` is thrown.
    pub fn create_image(
        &self,
        image_info: &erupt::vk::ImageCreateInfo,
        allocation_info: &AllocationCreateInfo,
    ) -> Result<(erupt::vk::Image, Allocation, AllocationInfo)> {
        let image_create_info = unsafe {
            mem::transmute::<erupt::vk::ImageCreateInfo, ffi::VkImageCreateInfo>(*image_info)
        };
        let allocation_create_info = allocation_create_info_to_ffi(allocation_info);
        let mut image: ffi::VkImage = std::ptr::null_mut();
        let mut allocation: Allocation = Default::default();
        let mut allocation_info: AllocationInfo = Default::default();
        let result = ffi_to_result(unsafe {
            ffi::vmaCreateImage(
                self.internal,
                &image_create_info,
                &allocation_create_info,
                &mut image,
                &mut allocation.internal,
                &mut allocation_info.internal,
            )
        });
        match result {
            erupt::vk::Result::SUCCESS => {
                Ok((erupt::vk::Image(image as u64), allocation, allocation_info))
            }
            _ => Err(Error::vulkan(result)),
        }
    }

    /// Destroys Vulkan image and frees allocated memory.
    ///
    /// This is just a convenience function equivalent to:
    ///
    /// ```ignore
    /// erupt::vk::Device::destroy_image(image, None);
    /// Allocator::free_memory(allocator, allocation);
    /// ```
    ///
    /// It it safe to pass null as `image` and/or `allocation`.
    pub fn destroy_image(&self, image: erupt::vk::Image, allocation: &Allocation) {
        unsafe {
            ffi::vmaDestroyImage(
                self.internal,
                image.object_handle() as ffi::VkImage,
                allocation.internal,
            );
        }
    }

    /// Destroys the internal allocator instance. After this has been called,
    /// no other functions may be called. Useful for ensuring a specific destruction
    /// order (for example, if an Allocator is a member of something that owns the Vulkan
    /// instance and destroys it in its own Drop).
    pub fn destroy(&mut self) {
        if !self.internal.is_null() {
            unsafe {
                ffi::vmaDestroyAllocator(self.internal);
                self.internal = std::ptr::null_mut();
            }
        }
    }
}

/// Custom `Drop` implementation to clean up internal allocation instance
impl Drop for Allocator {
    fn drop(&mut self) {
        self.destroy();
    }
}
