/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef NVP_SUPPORTS_OPTIX7


#include <sstream>

#include "vulkan/vulkan.h"

#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "denoiser.hpp"

#include "imgui/imgui_helper.h"
#include "nvvk/commands_vk.hpp"
#include "stb_image_write.h"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"


#include "_autogen/cpy_to_img.comp.h"
#include "_autogen/cpy_to_buffer.comp.h"


// Choose how to transfer images: 1 for a faster compute shader,
// or 0 value to use a simpler Vulkan command. The Vulkan way is simpler
// but about 5 times slower than the compute shader.
#define USE_COMPUTE_SHADER_TO_COPY 1

#define GRID_SIZE 16
inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}


DenoiserOptix::DenoiserOptix(nvvk::Context* ctx)
{
  setup(ctx->m_device, ctx->m_physicalDevice, ctx->m_queueGCT.familyIndex);
}

//--------------------------------------------------------------------------------------------------
// The denoiser will take an image, will convert it to a buffer (compatible with Cuda)
// will denoise the buffer, and put back the buffer to an image.
//
// To make this working, it is important that the VkDeviceMemory associated with the buffer
// has the 'export' functionality.


void DenoiserOptix::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueIndex)
{
  m_queueIndex     = queueIndex;
  m_device         = device;
  m_physicalDevice = physicalDevice;

  m_memAlloc.init(device, physicalDevice);
  m_allocEx.init(device, physicalDevice, &m_memAlloc);
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Initializing OptiX and creating the Denoiser instance
//
bool DenoiserOptix::initOptiX(const OptixDenoiserOptions& options, OptixPixelFormat pixelFormat, bool hdr)
{
  // Initialize CUDA
  CUDA_CHECK(cudaFree(nullptr));

  CUcontext cu_ctx = nullptr;  // zero means take the current context
  OPTIX_CHECK(optixInit());

  OptixDeviceContextOptions optixoptions = {};
  optixoptions.logCallbackFunction       = &contextLogCb;
  optixoptions.logCallbackLevel          = 4;

  OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &optixoptions, &m_optixDevice));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixDevice, contextLogCb, nullptr, 4));

  m_pixelFormat = pixelFormat;
  switch(pixelFormat)
  {

    case OPTIX_PIXEL_FORMAT_FLOAT3:
      m_sizeofPixel   = static_cast<uint32_t>(3 * sizeof(float));
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_FLOAT4:
      m_sizeofPixel = static_cast<uint32_t>(4 * sizeof(float));
#if OPTIX_VERSION >= 80000
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
#else
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR3:
      m_sizeofPixel   = static_cast<uint32_t>(3 * sizeof(uint8_t));
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR4:
      m_sizeofPixel = static_cast<uint32_t>(4 * sizeof(uint8_t));
#if OPTIX_VERSION >= 80000
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
#else
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
      break;
    case OPTIX_PIXEL_FORMAT_HALF3:
      m_sizeofPixel   = static_cast<uint32_t>(3 * sizeof(uint16_t));
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_HALF4:
      m_sizeofPixel = static_cast<uint32_t>(4 * sizeof(uint16_t));
#if OPTIX_VERSION >= 80000
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
#else
      m_denoiserAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
      break;
    default:
      assert(!"unsupported");
      break;
  }


  // This is to use RGB + Albedo + Normal
  m_denoiserOptions                 = options;
  OptixDenoiserModelKind model_kind = hdr ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
  model_kind                        = OPTIX_DENOISER_MODEL_KIND_AOV;
  OPTIX_CHECK(optixDenoiserCreate(m_optixDevice, model_kind, &m_denoiserOptions, &m_denoiser));


  return true;
}

//--------------------------------------------------------------------------------------------------
// Denoising the image in input and saving the denoised image in the output
//
void DenoiserOptix::denoiseImageBuffer(uint64_t& fenceValue, float blendFactor /*= 0.0f*/)
{
  try
  {
    OptixPixelFormat pixel_format        = m_pixelFormat;
    auto             sizeof_pixel        = m_sizeofPixel;
    uint32_t         row_stride_in_bytes = sizeof_pixel * m_imageSize.width;

    //std::vector<OptixImage2D> inputLayer;  // Order: RGB, Albedo, Normal

    // Create and set our OptiX layers
    OptixDenoiserLayer layer = {};
    // Input
    layer.input.data               = (CUdeviceptr)m_pixelBufferIn[0].cudaPtr;
    layer.input.width              = m_imageSize.width;
    layer.input.height             = m_imageSize.height;
    layer.input.rowStrideInBytes   = row_stride_in_bytes;
    layer.input.pixelStrideInBytes = m_sizeofPixel;
    layer.input.format             = pixel_format;

    // Output
    layer.output.data               = (CUdeviceptr)m_pixelBufferOut.cudaPtr;
    layer.output.width              = m_imageSize.width;
    layer.output.height             = m_imageSize.height;
    layer.output.rowStrideInBytes   = row_stride_in_bytes;
    layer.output.pixelStrideInBytes = sizeof(float) * 4;
    layer.output.format             = pixel_format;


    OptixDenoiserGuideLayer guide_layer = {};
    // albedo
    if(m_denoiserOptions.guideAlbedo != 0u)
    {
      guide_layer.albedo.data               = (CUdeviceptr)m_pixelBufferIn[1].cudaPtr;
      guide_layer.albedo.width              = m_imageSize.width;
      guide_layer.albedo.height             = m_imageSize.height;
      guide_layer.albedo.rowStrideInBytes   = row_stride_in_bytes;
      guide_layer.albedo.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.albedo.format             = pixel_format;
    }

    // normal
    if(m_denoiserOptions.guideNormal != 0u)
    {
      guide_layer.normal.data               = (CUdeviceptr)m_pixelBufferIn[2].cudaPtr;
      guide_layer.normal.width              = m_imageSize.width;
      guide_layer.normal.height             = m_imageSize.height;
      guide_layer.normal.rowStrideInBytes   = row_stride_in_bytes;
      guide_layer.normal.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.normal.format             = pixel_format;
    }

    // Wait from Vulkan (Copy to Buffer)
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags              = 0;
    wait_params.params.fence.value = fenceValue;
    cudaWaitExternalSemaphoresAsync(&m_semaphore.cu, &wait_params, 1, nullptr);

    if(m_dIntensity != 0)
    {
      OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_cuStream, &layer.input, m_dIntensity, m_dScratchBuffer,
                                                m_denoiserSizes.withoutOverlapScratchSizeInBytes));
    }

    OptixDenoiserParams denoiser_params{};
#if OPTIX_VERSION < 80000
    denoiser_params.denoiseAlpha = m_denoiserAlpha;
#endif
    denoiser_params.hdrIntensity = m_dIntensity;
    denoiser_params.blendFactor  = blendFactor;


    // Execute the denoiser
    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_cuStream, &denoiser_params, m_dStateBuffer,
                                    m_denoiserSizes.stateSizeInBytes, &guide_layer, &layer, 1, 0, 0, m_dScratchBuffer,
                                    m_denoiserSizes.withoutOverlapScratchSizeInBytes));


    CUDA_CHECK(cudaDeviceSynchronize());  // Making sure the denoiser is done
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    cudaExternalSemaphoreSignalParams sig_params{};
    sig_params.flags              = 0;
    sig_params.params.fence.value = ++fenceValue;
    cudaSignalExternalSemaphoresAsync(&m_semaphore.cu, &sig_params, 1, m_cuStream);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting all images to buffers used by the denoiser
//
void DenoiserOptix::imageToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn)
{
  LABEL_SCOPE_VK(cmdBuf);

#if USE_COMPUTE_SHADER_TO_COPY
  copyImageToBuffer(cmdBuf, imgIn);
#else
  VkBufferImageCopy region = {
      .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .imageExtent      = {.width = m_imageSize.width, .height = m_imageSize.height, .depth = 1},
  };

  for(int i = 0; i < static_cast<int>(imgIn.size()); i++)
  {
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_pixelBufferIn[i].bufVk.buffer, 1, &region);
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }
#endif
}


//--------------------------------------------------------------------------------------------------
// Converting the output buffer to the image
//
void DenoiserOptix::bufferToImage(const VkCommandBuffer& cmdBuf, nvvk::Texture* imgOut)
{
  LABEL_SCOPE_VK(cmdBuf);

#if USE_COMPUTE_SHADER_TO_COPY
  copyBufferToImage(cmdBuf, imgOut);
#else
  VkBufferImageCopy region = {
      .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
      .imageExtent      = {.width = m_imageSize.width, .height = m_imageSize.height, .depth = 1},
  };

  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  vkCmdCopyBufferToImage(cmdBuf, m_pixelBufferOut.bufVk.buffer, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
#endif
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroy()
{
  // Cleanup resources
  optixDenoiserDestroy(m_denoiser);
  optixDeviceContextDestroy(m_optixDevice);

  vkDestroySemaphore(m_device, m_semaphore.vk, nullptr);
  m_semaphore.vk = VK_NULL_HANDLE;

  destroyBuffer();
  for(auto& d : m_desc)
  {
    vkDestroyDescriptorPool(m_device, d.pool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, d.layout, nullptr);
    d.pool   = VK_NULL_HANDLE;
    d.layout = VK_NULL_HANDLE;
  }
  for(auto& p : m_pipelines)
  {
    vkDestroyPipeline(m_device, p.p, nullptr);
    vkDestroyPipelineLayout(m_device, p.layout, nullptr);
    p.p      = VK_NULL_HANDLE;
    p.layout = VK_NULL_HANDLE;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroyBuffer()
{
  for(auto& p : m_pixelBufferIn)
    p.destroy(m_allocEx);
  m_pixelBufferOut.destroy(m_allocEx);

  if(m_dStateBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dStateBuffer));
    m_dStateBuffer = 0;
  }
  if(m_dScratchBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dScratchBuffer));
    m_dScratchBuffer = 0;
  }
  if(m_dIntensity != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dIntensity));
    m_dIntensity = 0;
  }
  if(m_dMinRGB != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dMinRGB));
    m_dMinRGB = 0;
  }
}

//--------------------------------------------------------------------------------------------------
// UI specific for the denoiser
//
bool DenoiserOptix::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
  {
    modified |= ImGuiH::Control::Checkbox("Denoise", "", reinterpret_cast<bool*>(&m_denoisedMode));
    modified |= ImGuiH::Control::Slider("Start Frame", "Frame at which the denoiser starts to be applied",
                                        &m_startDenoiserFrame, nullptr, ImGuiH::Control::Flags::Normal, 0, 99);
  }
  return modified;
}

//--------------------------------------------------------------------------------------------------
// Allocating all the buffers in which the images will be transfered.
// The buffers are shared with Cuda, therefore OptiX can denoised them
//
void DenoiserOptix::allocateBuffers(const VkExtent2D& imgSize)
{
  m_imageSize = imgSize;

  destroyBuffer();

  VkDeviceSize buffer_size = static_cast<VkDeviceSize>(m_imageSize.width) * m_imageSize.height * 4 * sizeof(float);
  VkBufferUsageFlags usage{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

  {  // Color
    m_pixelBufferIn[0].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[0]);  // Exporting the buffer to Cuda handle and pointers
    NAME_VK(m_pixelBufferIn[0].bufVk.buffer);
  }

  // Albedo
  {
    m_pixelBufferIn[1].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[1]);
    NAME_VK(m_pixelBufferIn[1].bufVk.buffer);
  }
  // Normal
  {
    m_pixelBufferIn[2].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[2]);
    NAME_VK(m_pixelBufferIn[2].bufVk.buffer);
  }

  // Output image/buffer
  m_pixelBufferOut.bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  createBufferCuda(m_pixelBufferOut);
  NAME_VK(m_pixelBufferOut.bufVk.buffer);


  // Computing the amount of memory needed to do the denoiser
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageSize.width, m_imageSize.height, &m_denoiserSizes));

  CUDA_CHECK(cudaMalloc((void**)&m_dStateBuffer, m_denoiserSizes.stateSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dScratchBuffer, m_denoiserSizes.withoutOverlapScratchSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dMinRGB, 4 * sizeof(float)));
  if(m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT3 || m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT4)
    CUDA_CHECK(cudaMalloc((void**)&m_dIntensity, sizeof(float)));

  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_cuStream, m_imageSize.width, m_imageSize.height, m_dStateBuffer,
                                 m_denoiserSizes.stateSizeInBytes, m_dScratchBuffer, m_denoiserSizes.withoutOverlapScratchSizeInBytes));
}


//--------------------------------------------------------------------------------------------------
// Get the Vulkan buffer and create the Cuda equivalent using the memory allocated in Vulkan
//
void DenoiserOptix::createBufferCuda(BufferCuda& buf)
{
  nvvk::MemAllocator::MemInfo mem_info = m_allocEx.getMemoryAllocator()->getMemoryInfo(buf.bufVk.memHandle);
#ifdef WIN32
  VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = mem_info.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  vkGetMemoryWin32HandleKHR(m_device, &info, &buf.handle);
#else
  VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = mem_info.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  vkGetMemoryFdKHR(m_device, &info, &buf.handle);
#endif

  VkMemoryRequirements memory_req{};
  vkGetBufferMemoryRequirements(m_device, buf.bufVk.buffer, &memory_req);

  cudaExternalMemoryHandleDesc cuda_ext_mem_handle_desc{};
  cuda_ext_mem_handle_desc.size = memory_req.size;
#ifdef WIN32
  cuda_ext_mem_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
  cuda_ext_mem_handle_desc.handle.win32.handle = buf.handle;
#else
  cuda_ext_mem_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
  cuda_ext_mem_handle_desc.handle.fd = buf.handle;
#endif

  cudaExternalMemory_t cuda_ext_mem_vertex_buffer{};
  CUDA_CHECK(cudaImportExternalMemory(&cuda_ext_mem_vertex_buffer, &cuda_ext_mem_handle_desc));

#ifndef WIN32
  // fd got consumed
  cuda_ext_mem_handle_desc.handle.fd = -1;
#endif

  cudaExternalMemoryBufferDesc cuda_ext_buffer_desc{};
  cuda_ext_buffer_desc.offset = 0;
  cuda_ext_buffer_desc.size   = memory_req.size;
  cuda_ext_buffer_desc.flags  = 0;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr, cuda_ext_mem_vertex_buffer, &cuda_ext_buffer_desc));
}

//--------------------------------------------------------------------------------------------------
// Creating the timeline semaphores for syncing with CUDA
//
void DenoiserOptix::createSemaphore()
{
#ifdef WIN32
  auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

  VkSemaphoreTypeCreateInfo timeline_create_info{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                 .semaphoreType = VK_SEMAPHORE_TYPE_BINARY,
                                                 .initialValue  = 0};

  VkExportSemaphoreCreateInfo esci{.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
                                   .pNext       = &timeline_create_info,
                                   .handleTypes = VkExternalSemaphoreHandleTypeFlags(handle_type)};

  VkSemaphoreCreateInfo sci{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esci};

  vkCreateSemaphore(m_device, &sci, nullptr, &m_semaphore.vk);

#ifdef WIN32
  VkSemaphoreGetWin32HandleInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  handle_info.handleType = handle_type;
  handle_info.semaphore  = m_semaphore.vk;
  vkGetSemaphoreWin32HandleKHR(m_device, &handle_info, &m_semaphore.handle);
#else
  VkSemaphoreGetFdInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  handle_info.handleType = handle_type;
  handle_info.semaphore  = m_semaphore.vk;
  vkGetSemaphoreFdKHR(m_device, &handle_info, &m_semaphore.handle);
#endif


  cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc{};
  std::memset(&external_semaphore_handle_desc, 0, sizeof(external_semaphore_handle_desc));
  external_semaphore_handle_desc.flags = 0;
#ifdef WIN32
  external_semaphore_handle_desc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  external_semaphore_handle_desc.handle.win32.handle = static_cast<void*>(m_semaphore.handle);
#else
  external_semaphore_handle_desc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
  external_semaphore_handle_desc.handle.fd = m_semaphore.handle;
#endif

  CUDA_CHECK(cudaImportExternalSemaphore(&m_semaphore.cu, &external_semaphore_handle_desc));
}

///
/// The second below is for the compute shaders, which copies images to buffers, or buffer to image
///
void DenoiserOptix::createCopyPipeline()
{
  {
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[eCpyToBuffer].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[eCpyToBuffer].layout, bind.createLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));

    // Pipeline
    VkPipelineLayoutCreateInfo pipe_info{
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts    = &m_desc[eCpyToBuffer].layout,
    };
    vkCreatePipelineLayout(m_device, &pipe_info, nullptr, &m_pipelines[eCpyToBuffer].layout);
    NAME_VK(m_pipelines[eCpyToBuffer].layout);

    VkPipelineShaderStageCreateInfo stage_info{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = nvvk::createShaderModule(m_device, cpy_to_buffer_comp, sizeof(cpy_to_buffer_comp)),
        .pName  = "main",
    };

    VkComputePipelineCreateInfo comp_info{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = stage_info,
        .layout = m_pipelines[eCpyToBuffer].layout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_pipelines[eCpyToBuffer].p);
    NAME_VK(m_pipelines[eCpyToBuffer].p);

    vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
  }

  {
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[eCpyToImage].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[eCpyToImage].layout, bind.createLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));

    // Pipeline
    VkPipelineLayoutCreateInfo pipe_info{
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts    = &m_desc[eCpyToImage].layout,
    };
    vkCreatePipelineLayout(m_device, &pipe_info, nullptr, &m_pipelines[eCpyToImage].layout);
    NAME_VK(m_pipelines[eCpyToImage].layout);

    VkPipelineShaderStageCreateInfo stage_info{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = nvvk::createShaderModule(m_device, cpy_to_img_comp, sizeof(cpy_to_img_comp)),
        .pName  = "main",
    };

    VkComputePipelineCreateInfo comp_info{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = stage_info,
        .layout = m_pipelines[eCpyToImage].layout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_pipelines[eCpyToImage].p);
    NAME_VK(m_pipelines[eCpyToImage].p);

    vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
  }
}

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorImageInfo* img)
{
  VkWriteDescriptorSet wrt{
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet          = set,
      .dstBinding      = bind,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo      = img,
  };
  return wrt;
};

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorBufferInfo* buf)
{
  VkWriteDescriptorSet wrt{
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet          = set,
      .dstBinding      = bind,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo     = buf,
  };
  return wrt;
};

//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyImageToBuffer
//
void DenoiserOptix::copyImageToBuffer(const VkCommandBuffer& cmd, const std::vector<nvvk::Texture>& imgIn)
{
  LABEL_SCOPE_VK(cmd);

  VkDescriptorImageInfo  img0 = imgIn[0].descriptor;
  VkDescriptorImageInfo  img1 = imgIn[1].descriptor;
  VkDescriptorImageInfo  img2 = imgIn[2].descriptor;
  VkDescriptorBufferInfo buf0 = {.buffer = m_pixelBufferIn[0].bufVk.buffer, .range = VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf1 = {.buffer = m_pixelBufferIn[1].bufVk.buffer, .range = VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf2 = {.buffer = m_pixelBufferIn[2].bufVk.buffer, .range = VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite({}, 0, &img0));
  writes.emplace_back(makeWrite({}, 1, &img1));
  writes.emplace_back(makeWrite({}, 2, &img2));
  writes.emplace_back(makeWrite({}, 3, &buf0));
  writes.emplace_back(makeWrite({}, 4, &buf1));
  writes.emplace_back(makeWrite({}, 5, &buf2));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[eCpyToBuffer].layout, 0,
                            static_cast<uint32_t>(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[eCpyToBuffer].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}


//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyBufferToImage
//
void DenoiserOptix::copyBufferToImage(const VkCommandBuffer& cmd, const nvvk::Texture* imgIn)
{
  VkDescriptorImageInfo  img0 = imgIn->descriptor;
  VkDescriptorBufferInfo buf0 = {.buffer = m_pixelBufferOut.bufVk.buffer, .range = VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite({}, 0, &img0));
  writes.emplace_back(makeWrite({}, 1, &buf0));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[eCpyToImage].layout, 0,
                            static_cast<uint32_t>(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[eCpyToImage].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}


#endif  // !NVP_SUPPORTS_OPTIX7
