/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

 This sample load GLTF scenes and render using RTX (path tracer)
 
 The path tracer is rendering in multiple G-Buffers, which are used 
 for denoising the "result". 

 The final image will be tonemapped, either the "result" or the "denoised" 
 and it is the tonemmaped Ldr image that is displayed.

 Look for #OPTIX_D, to find what was added to this example to enable the 
 OptiX denoiser.

 Note: regarding semaphores, after raytracing a Vulkan semaphore is emitted
       a the Cuda side waits for its result. When Cuda is done, it sends a
       semaphore as well and this one is added to the Application frame
       wait semaphore. Therefore, CPU isn't been blocked and further Vulkan 
       commands can be filled, but the last portion of it, won't be executed 
       until the Cuda denoiser is finished. (See m_app->addWaitSemaphore())

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <filesystem>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "imgui/imgui_axis.hpp"
#include "nvh/fileoperations.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/hdr_env_dome.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "denoiser.hpp"


#include "shaders/device_host.h"
#include "shaders/dh_bindings.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/gbuffers.rchit.h"
#include "_autogen/gbuffers.rmiss.h"
#include "nvvkhl/element_benchmark_parameters.hpp"


std::shared_ptr<nvvkhl::ElementCamera>              g_elemCamera;
std::shared_ptr<nvvkhl::ElementBenchmarkParameters> g_elemBenchmark;


namespace nvvkhl {
//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class OptixDenoiserEngine : public nvvkhl::IAppElement
{
  enum GbufferNames
  {
    eGBufLdr,
    eGBufResult,
    eGBufAlbedo,
    eGBufNormal,
    eGbufDenoised,
  };

  struct Settings
  {
    int       maxFrames{200000};
    int       maxSamples{1};
    int       maxDepth{5};
    bool      showAxis{true};
    glm::vec4 clearColor{1.F};
    float     envRotation{0.F};
    bool      denoiseApply{true};
    bool      denoiseFirstFrame{false};
    int       denoiseEveryNFrames{100};
  } m_settings;

public:
  OptixDenoiserEngine()
  {
    m_frameInfo.maxLuminance = 10.0F;
    m_frameInfo.clearColor   = glm::vec4(1.F);
  };

  ~OptixDenoiserEngine() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app            = app;
    m_device         = m_app->getDevice();
    m_physicalDevice = m_app->getPhysicalDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);                            // Debug utility
    m_alloc      = std::make_unique<AllocVma>(allocator_info);                             // Allocator
    m_scene      = std::make_unique<nvh::gltf::Scene>();                                   // GLTF scene
    m_sceneVk    = std::make_unique<SceneVk>(m_device, m_physicalDevice, m_alloc.get());   // GLTF Scene buffers
    m_sceneRtx   = std::make_unique<SceneRtx>(m_device, m_physicalDevice, m_alloc.get());  // GLTF Scene BLAS/TLAS
    m_tonemapper = std::make_unique<TonemapperPostProcess>(m_device, m_alloc.get());
    m_sbt        = std::make_unique<nvvk::SBTWrapper>();
    m_picker     = std::make_unique<nvvk::RayPickerKHR>(m_device, m_physicalDevice, m_alloc.get());
    m_hdrEnv     = std::make_unique<HdrEnv>(m_device, m_physicalDevice, m_alloc.get());
    m_rtxSet     = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_sceneSet   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Override the way benchmark count frames, to only use valid ones
    g_elemBenchmark->setCurrentFrame([&] { return m_frame; });

#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser = std::make_unique<DenoiserOptix>();
    m_denoiser->setup(m_device, m_physicalDevice, m_app->getQueue(0).familyIndex);

    OptixDenoiserOptions d_options;
    d_options.guideAlbedo = 1u;
    d_options.guideNormal = 1u;
    m_denoiser->initOptiX(d_options, OPTIX_PIXEL_FORMAT_FLOAT4, true);
    m_denoiser->createSemaphore();
    m_denoiser->createCopyPipeline();
#else
    m_settings.denoiseApply = false;
    LOGE("OptiX is not supported");
#endif  // NVP_SUPPORTS_OPTIX7


    m_hdrEnv->loadEnvironment("");

    // Requesting ray tracing properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
    // Create utilities to create the Shading Binding Table (SBT)
    uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt->setup(m_app->getDevice(), gct_queue_index, m_alloc.get(), rt_prop);

    // Create resources
    createCommandBuffers();
    createGbuffers(m_viewSize);
    createVulkanBuffers();

    m_tonemapper->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    // Tonemapper is using GBuffer-1 as input and output to GBuffer-0
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eGBufResult),
                                              m_gBuffers->getDescriptorImageInfo(eGBufLdr));
    writeRtxSet();
  }

  void onUIMenu() override
  {
    bool load_file{false};

    windowTitle();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Load", "Ctrl+O"))
      {
        load_file = true;
      }
      ImGui::Separator();
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      load_file = true;
    }

    if(load_file)
    {
      auto filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                      "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const char* filename) override
  {
    namespace fs = std::filesystem;
    vkDeviceWaitIdle(m_device);
    std::string extension = fs::path(filename).extension().string();
    if(extension == ".gltf" || extension == ".glb")
    {
      createScene(filename);
    }
    else if(extension == ".hdr")
    {
      createHdr(filename);
      resetFrame();
    }


    resetFrame();
  }

  void onUIRender() override
  {
    using namespace ImGuiH;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_M))
    {
      onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
      reset = true;
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        ImGuiH::CameraWidget();
      }

      if(ImGui::CollapsingHeader("Settings"))
      {
        PropertyEditor::begin();

        if(PropertyEditor::treeNode("Ray Tracing"))
        {
          reset |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 10); });
          reset |= PropertyEditor::entry("Samples", [&] { return ImGui::SliderInt("#2", &m_settings.maxSamples, 1, 5); });
          reset |= PropertyEditor::entry("Frames",
                                         [&] { return ImGui::DragInt("#3", &m_settings.maxFrames, 5.0F, 1, 1000000); });
          PropertyEditor::treePop();
        }
        PropertyEditor::entry("Show Axis", [&] { return ImGui::Checkbox("##4", &m_settings.showAxis); });
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Environment"))
      {
        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Hdr"))
        {
          reset |= PropertyEditor::entry(
              "Color", [&] { return ImGui::ColorEdit3("##Color", &m_settings.clearColor.x, ImGuiColorEditFlags_Float); },
              "Color multiplier");

          reset |= PropertyEditor::entry(
              "Rotation", [&] { return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        m_tonemapper->onUI();
      }

      // #OPTIX_D
      if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
      {
        ImGui::Checkbox("Denoise", &m_settings.denoiseApply);
        ImGui::Checkbox("First Frame", &m_settings.denoiseFirstFrame);
        ImGui::SliderInt("N-frames", &m_settings.denoiseEveryNFrames, 1, 500);
        ImGui::SliderFloat("Blend", &m_blendFactor, 0.f, 1.0f);
        int denoised_frame = -1;
        if(m_settings.denoiseApply)
        {
          if(m_frame >= m_settings.maxFrames)
            denoised_frame = m_settings.maxFrames;
          else if(m_settings.denoiseFirstFrame && (m_frame < m_settings.denoiseEveryNFrames))
            denoised_frame = 0;
          else if(m_frame >= m_settings.denoiseEveryNFrames)
            denoised_frame = (m_frame / m_settings.denoiseEveryNFrames) * m_settings.denoiseEveryNFrames;
        }
        ImGui::Text("Denoised Frame: %d", denoised_frame);

        ImVec2 tumbnailSize = {150 * m_gBuffers->getAspectRatio(), 150};
        ImGui::Text("Albedo");
        ImGui::Image(m_gBuffers->getDescriptorSet(eGBufAlbedo), tumbnailSize);
        ImGui::Text("Normal");
        ImGui::Image(m_gBuffers->getDescriptorSet(eGBufNormal), tumbnailSize);
        ImGui::Text("Result");
        ImGui::Image(m_gBuffers->getDescriptorSet(eGBufResult), tumbnailSize);
        ImGui::Text("Denoised");
        ImGui::Image(m_gBuffers->getDescriptorSet(eGbufDenoised), tumbnailSize);
      }

      ImGui::End();

      if(reset)
      {
        resetFrame();
      }
    }

    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(showDenoisedImage() ? eGbufDenoised : eGBufResult),
                                              m_gBuffers->getDescriptorImageInfo(eGBufLdr));


    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(eGBufLdr), ImGui::GetContentRegionAvail());

      if(m_settings.showAxis)
      {  // Display orientation axis at the bottom left corner of the window
        const float axisSize = 25.F;
        ImVec2      pos      = ImGui::GetWindowPos();
        pos.y += ImGui::GetWindowSize().y;
        pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
        ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
      }

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer /*cmd*/) override
  {
    if(!m_scene->valid())
      return;
    // Update the frame only if the scene is valid
    if(!updateFrame())
      return;

    // Using local command buffer for the frame
    const CommandFrame& commandFrame = m_commandFrames[m_app->getFrameCycleIndex()];
    VkCommandBuffer     cmd          = commandFrame.cmdBuffer[0];
    vkResetCommandPool(m_device, commandFrame.cmdPool, 0);
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd, &begin_info);

    // Get camera info
    float     view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    const auto& clip = CameraManip.getClipPlanes();
    m_frameInfo.view = CameraManip.getMatrix();
    m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
    m_frameInfo.proj[1][1] *= -1;
    m_frameInfo.projInv     = glm::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv     = glm::inverse(m_frameInfo.view);
    m_frameInfo.camPos      = eye;
    m_frameInfo.envRotation = m_settings.envRotation;
    m_frameInfo.clearColor  = m_settings.clearColor;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.maxSamples = m_settings.maxSamples;
    m_pushConst.frame      = m_frame;

    raytraceScene(cmd);

#ifdef NVP_SUPPORTS_OPTIX7
    // #OPTIX_D
    if(needToDenoise())
    {
      // Submit raytracing and signal
      copyImagesToCuda(cmd);
      vkEndCommandBuffer(cmd);  // Need to end the command buffer to submit the semaphore

      // Prepare the signal semaphore for the OptiX denoiser
      VkSemaphoreSubmitInfoKHR signal_semaphore{
          .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
          .semaphore = m_denoiser->getTLSemaphore(),
          .value     = ++m_fenceValue,  // Increment for signaling
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
      };

      VkCommandBufferSubmitInfoKHR cmd_buf_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, 0, cmd};

      VkSubmitInfo2KHR submits{
          .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
          .commandBufferInfoCount   = 1,
          .pCommandBufferInfos      = &cmd_buf_info,
          .signalSemaphoreInfoCount = 1,
          .pSignalSemaphoreInfos    = &signal_semaphore,
      };

      // Submit rendering and signal when done
      vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submits, {});

      // #OPTIX_D
      // Denoiser waits for signal (Vulkan) and submit (Cuda) new one when done
      denoiseImage();

      // #OPTIX_D
      // Adding a wait semaphore to the application, such that the frame command buffer,
      // will wait for the end of the denoised image before executing the command buffer.
      VkSemaphoreSubmitInfo wait_semaphore{
          .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
          .semaphore = m_denoiser->getTLSemaphore(),
          .value     = m_fenceValue,
          .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      };
      m_app->addWaitSemaphore(wait_semaphore);


      // #OPTIX_D
      // Continue rendering pipeline (using the second command buffer)
      cmd = commandFrame.cmdBuffer[1];

      VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      vkBeginCommandBuffer(cmd, &begin_info);
      copyCudaImagesToVulkan(cmd);
    }
#endif

    // Apply tonemapper - take GBuffer-X and output to GBuffer-0
    m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

    // End of the first or second command buffer
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo submit_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR};
    submit_info.commandBuffer = cmd;
    m_app->prependCommandBuffer(submit_info);  // Prepend to the frame command buffer
  }


private:
  void createScene(const std::string& filename)
  {
    m_scene->load(filename);
    nvvkhl::setCamera(filename, m_scene->getRenderCameras(), m_scene->getSceneBounds());  // Camera auto-scene-fitting
    g_elemCamera->setSceneRadius(m_scene->getSceneBounds().radius());                     // Navigation help

    {  // Create the Vulkan side of the scene
      auto cmd = m_app->createTempCmdBuffer();
      m_sceneVk->create(cmd, *m_scene);
      m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Create BLAS / TLAS
      m_app->submitAndWaitTempCmdBuffer(cmd);

      m_picker->setTlas(m_sceneRtx->tlas());
    }

    m_allNodes      = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterAll);
    m_solidMatNodes = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterSolid);
    m_blendMatNodes = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterBlend);

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
  }


  void createGbuffers(const glm::vec2& size)
  {
    static auto depth_format = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

    m_viewSize = size;
    VkExtent2D vk_size{static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};

    // Four GBuffers: RGBA8 and 4x RGBA32F(final,albedo,normal, denoised), rendering to RGBA32F and tone mapped to RGBA8
    std::vector<VkFormat> color_buffers = {
        VK_FORMAT_R8G8B8A8_UNORM,       // LDR
        VK_FORMAT_R32G32B32A32_SFLOAT,  // Result
        VK_FORMAT_R32G32B32A32_SFLOAT,  // Albedo
        VK_FORMAT_R32G32B32A32_SFLOAT,  // Normal
        VK_FORMAT_R32G32B32A32_SFLOAT,  // Denoised
    };

    // Creation of the GBuffers
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, color_buffers, depth_format);

#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser->allocateBuffers(vk_size);
#endif

    // Indicate the renderer to reset its frame
    resetFrame();
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createRtxSet()
  {
    auto& d = m_rtxSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    // #OPTIX_D
    d->addBinding(RtxBindings::eOutAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(RtxBindings::eOutNormal, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  void createSceneSet()
  {
    auto& d = m_sceneSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtxPipe;
    p.destroy(m_device);
    p.plines.resize(1);

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eMissGbuf,  // #OPTIX_D
      eClosestHit,
      eAnyHit,
      eClosestHitGbuf,  // #OPTIX_D
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");
    // AnyHit
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
    stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit] = stage;
    m_dutil->setObjectName(stage.module, "Any Hit");
    // #OPTIX_D
    // Miss G-Buffers
    stage.module      = nvvk::createShaderModule(m_device, gbuffers_rmiss, sizeof(gbuffers_rmiss));
    stage.stage       = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMissGbuf] = stage;
    // Hit Group - Closest Hit
    stage.module            = nvvk::createShaderModule(m_device, gbuffers_rchit, sizeof(gbuffers_rchit));
    stage.stage             = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHitGbuf] = stage;

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // #OPTIX_D
    // Miss - G-Buf
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMissGbuf;
    shader_groups.push_back(group);

    // Hit Group-0
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    shader_groups.push_back(group);

    // #OPTIX_D
    // Hit Group-1
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHitGbuf;
    group.anyHitShader     = VK_SHADER_UNUSED_KHR;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtxSet->getLayout(), m_sceneSet->getLayout(),
                                                              m_hdrEnv->getDescriptorSetLayout()};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups                      = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data());
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt->create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void writeRtxSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_rtxSet;

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;
    VkDescriptorImageInfo image_info{{}, m_gBuffers->getColorImageView(eGBufResult), VK_IMAGE_LAYOUT_GENERAL};
    // #OPTIX_D
    VkDescriptorImageInfo albedo_info{{}, m_gBuffers->getColorImageView(eGBufAlbedo), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo normal_info{{}, m_gBuffers->getColorImageView(eGBufNormal), VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
    writes.emplace_back(d->makeWrite(0, RtxBindings::eOutImage, &image_info));
    // #OPTIX_D
    writes.emplace_back(d->makeWrite(0, RtxBindings::eOutAlbedo, &albedo_info));
    writes.emplace_back(d->makeWrite(0, RtxBindings::eOutNormal, &normal_info));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }


  void writeSceneSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_sceneSet;

    // Write to descriptors
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo scene_desc{m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, SceneBindings::eFrameInfo, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk->textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(d->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static glm::mat4 ref_cam_matrix;
    static float     ref_fov{CameraManip.getFov()};

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

    if(ref_cam_matrix != m || ref_fov != fov)
    {
      resetFrame();
      ref_cam_matrix = m;
      ref_fov        = fov;
    }

    if(m_frame >= m_settings.maxFrames)
    {
      return false;
    }
    m_frame++;
    return true;
  }

  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = -1; }

  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const auto&           size = m_app->getViewportSize();
      std::array<char, 256> buf{};
      int ret = snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", PROJECT_NAME,
                         static_cast<int>(size.width), static_cast<int>(size.height),
                         static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate, m_frame);
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
      dirty_timer = 0;
    }
  }


  //--------------------------------------------------------------------------------------------------
  // Send a ray under mouse coordinates, and retrieve the information
  // - Set new camera interest point on hit position
  //
  void screenPicking()
  {
    auto* tlas = m_sceneRtx->tlas();
    if(tlas == VK_NULL_HANDLE)
      return;

    ImGui::Begin("Viewport");  // ImGui, picking within "viewport"
    auto  mouse_pos        = ImGui::GetMousePos();
    auto  main_size        = ImGui::GetContentRegionAvail();
    auto  corner           = ImGui::GetCursorScreenPos();  // Corner of the viewport
    float aspect_ratio     = main_size.x / main_size.y;
    mouse_pos              = mouse_pos - corner;
    ImVec2 local_mouse_pos = mouse_pos / main_size;
    ImGui::End();

    auto* cmd = m_app->createTempCmdBuffer();

    // Finding current camera matrices
    const auto& view = CameraManip.getMatrix();
    auto        proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPickerKHR::PickInfo pick_info;
    pick_info.pickX          = local_mouse_pos.x;
    pick_info.pickY          = local_mouse_pos.y;
    pick_info.modelViewInv   = glm::inverse(view);
    pick_info.perspectiveInv = glm::inverse(proj);

    // Run and wait for result
    m_picker->run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Retrieving picking information
    nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
    if(pr.instanceID == ~0)
    {
      LOGI("Nothing Hit\n");
      return;
    }

    if(pr.hitT <= 0.F)
    {
      LOGI("Hit Distance == 0.0\n");
      return;
    }

    // Find where the hit point is and set the interest position
    glm::vec3 world_pos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, world_pos, up, false);

    //    auto float_as_uint = [](float f) { return *reinterpret_cast<uint32_t*>(&f); };

    // Logging picking info.
    const auto& renderNode = m_scene->getRenderNodes()[pr.instanceID];
    std::string name       = m_scene->getModel().nodes[renderNode.refNodeID].name;
    LOGI("Hit(%d): %s, PrimId: %d", pr.instanceCustomIndex, name.c_str(), pr.primitiveID);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
    LOGI("PrimitiveID: %d\n", pr.primitiveID);
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtxSet->getSet(), m_sceneSet->getSet(), m_hdrEnv->getDescriptorSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

    const auto& regions = m_sbt->getRegions();
    const auto& size    = m_gBuffers->getSize();
    vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

    // Making sure the rendered image is ready to be used
    {
      auto scope_dbg2 = m_dutil->scopeLabel(cmd, "barrier");

      auto image_memory_barrier =
          nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eGBufResult), VK_ACCESS_SHADER_READ_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &image_memory_barrier);
    }
  }

  void createHdr(const char* filename)
  {
    m_hdrEnv = std::make_unique<HdrEnv>(m_device, m_physicalDevice, m_alloc.get());

    m_hdrEnv->loadEnvironment(filename);
  }

  // #OPTIX_D
  // Return true if the current frame need to be denoised, as we are not  denoising all frames.
  bool needToDenoise() const
  {
    if(m_settings.denoiseApply)
    {
      if(m_frame == m_settings.maxFrames)
        return true;
      if(!m_settings.denoiseFirstFrame && m_frame == 0)
        return false;
      if(m_frame % m_settings.denoiseEveryNFrames == 0)
        return true;
    }
    return false;
  }

  // #OPTIX_D
  // Will copy the Vulkan images to Cuda buffers
  void copyImagesToCuda(VkCommandBuffer cmd)
  {
#ifdef NVP_SUPPORTS_OPTIX7
    nvvk::Texture result{m_gBuffers->getColorImage(eGBufResult), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufResult)};
    nvvk::Texture albedo{m_gBuffers->getColorImage(eGBufAlbedo), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufAlbedo)};
    nvvk::Texture normal{m_gBuffers->getColorImage(eGBufNormal), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufNormal)};
    m_denoiser->imageToBuffer(cmd, {result, albedo, normal});
#endif  // NVP_SUPPORTS_OPTIX7
  }

  // #OPTIX_D
  // Copy the denoised buffer to Vulkan image
  void copyCudaImagesToVulkan(VkCommandBuffer cmd)
  {
#ifdef NVP_SUPPORTS_OPTIX7
    nvvk::Texture denoised{m_gBuffers->getColorImage(eGbufDenoised), nullptr, m_gBuffers->getDescriptorImageInfo(eGbufDenoised)};
    m_denoiser->bufferToImage(cmd, &denoised);
#endif  // NVP_SUPPORTS_OPTIX7
  }

  // #OPTIX_D
  // Invoke the Optix denoiser
  void denoiseImage()
  {
#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser->denoiseImageBuffer(m_fenceValue, m_blendFactor);
#endif  // NVP_SUPPORTS_OPTIX7
  }

  // #OPTIX_D
  // Determine which image will be displayed, the original from ray tracer or the denoised one
  bool showDenoisedImage() const
  {
    return m_settings.denoiseApply
           && ((m_frame >= m_settings.denoiseEveryNFrames) || m_settings.denoiseFirstFrame || (m_frame >= m_settings.maxFrames));
  }


  void createCommandBuffers()
  {
    // Max 3 frames in flight
    for(uint32_t i = 0; i < 3; i++)
    {
      CommandFrame* cf = &m_commandFrames[i];
      {
        VkCommandPoolCreateInfo info = {.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                        .flags            = 0,
                                        .queueFamilyIndex = m_app->getQueue(0).familyIndex};
        NVVK_CHECK(vkCreateCommandPool(m_device, &info, nullptr, &cf->cmdPool));
        m_dutil->setObjectName(cf->cmdPool, "Pool" + std::to_string(i));
      }
      {
        VkCommandBufferAllocateInfo info = {.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            .commandPool        = cf->cmdPool,
                                            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                            .commandBufferCount = 2};
        NVVK_CHECK(vkAllocateCommandBuffers(m_device, &info, cf->cmdBuffer));
        m_dutil->setObjectName(cf->cmdBuffer[0], fmt::format("Cmd[{}][0]", i));
        m_dutil->setObjectName(cf->cmdBuffer[1], fmt::format("Cmd[{}][1]", i));
      }
    }
  }

  void destroyResources()
  {
    m_alloc->destroy(m_bFrameInfo);

    for(auto& f : m_commandFrames)
    {
      vkFreeCommandBuffers(m_device, f.cmdPool, 2, f.cmdBuffer);
      vkDestroyCommandPool(m_device, f.cmdPool, nullptr);
    }
    m_gBuffers.reset();

    m_rasterPipe.destroy(m_device);
    m_rtxPipe.destroy(m_device);
    m_rtxSet->deinit();
    m_sceneSet->deinit();
    m_sbt->destroy();
    m_picker->destroy();
#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser->destroy();
#endif
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*             m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  std::unique_ptr<AllocVma>        m_alloc;

  glm::vec2                                     m_viewSize       = {1, 1};
  VkClearColorValue                             m_clearColor     = {{0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color
  VkDevice                                      m_device         = VK_NULL_HANDLE;              // Convenient
  VkPhysicalDevice                              m_physicalDevice = VK_NULL_HANDLE;              // Convenient
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;  // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet;    // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer> m_sceneSet;  // Descriptor set

  // Resources
  nvvk::Buffer m_bFrameInfo;

  // Pipeline
  PushConstant      m_pushConst{};  // Information sent to the shader
  PipelineContainer m_rasterPipe;
  PipelineContainer m_rtxPipe;
  int               m_frame{-1};
  FrameInfo         m_frameInfo{};

  std::unique_ptr<nvh::gltf::Scene>      m_scene;
  std::unique_ptr<SceneVk>               m_sceneVk;
  std::unique_ptr<SceneRtx>              m_sceneRtx;
  std::unique_ptr<TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<nvvk::SBTWrapper>      m_sbt;     // Shading binding table wrapper
  std::unique_ptr<nvvk::RayPickerKHR>    m_picker;  // For ray picking info
  std::unique_ptr<HdrEnv>                m_hdrEnv;

  // For rendering all nodes
  std::vector<uint32_t> m_solidMatNodes;
  std::vector<uint32_t> m_blendMatNodes;
  std::vector<uint32_t> m_allNodes;

#ifdef NVP_SUPPORTS_OPTIX7
  std::unique_ptr<DenoiserOptix> m_denoiser;
  uint64_t                       m_fenceValue{0U};
#endif  // NVP_SUPPORTS_OPTIX7
  float m_blendFactor = 0.0f;

  // Command buffers for rendering
  struct CommandFrame
  {
    VkCommandPool   cmdPool      = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuffer[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  };
  std::array<CommandFrame, 3> m_commandFrames;
};

}  // namespace nvvkhl


//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name  = PROJECT_NAME " Example";
  spec.vSync = true;

  nvvk::ContextCreateInfo vkSetup;
  vkSetup.apiMajor = 1;
  vkSetup.apiMinor = 3;

  vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &ray_query_features);  // Used for picking
  vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // #OPTIX_D
  // Semaphores - interop Vulkan/Cuda
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
#ifdef WIN32
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif

  // Synchronization (mix of timeline and binary semaphores)
  vkSetup.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);

  // Buffer - interop
  vkSetup.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);


  // Display extension
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Creating the Vulkan context
  auto m_context = std::make_shared<nvvk::Context>();
  m_context->init(vkSetup);

  // Application Vulkan setup
  spec.instance       = m_context->m_instance;
  spec.device         = m_context->m_device;
  spec.physicalDevice = m_context->m_physicalDevice;
  spec.queues.push_back({m_context->m_queueGCT.familyIndex, m_context->m_queueGCT.queueIndex, m_context->m_queueGCT.queue});
  spec.queues.push_back({m_context->m_queueC.familyIndex, m_context->m_queueC.queueIndex, m_context->m_queueC.queue});
  spec.queues.push_back({m_context->m_queueT.familyIndex, m_context->m_queueT.queueIndex, m_context->m_queueT.queue});


  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  g_elemBenchmark    = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);  // Benchmarking
  g_elemCamera       = std::make_shared<nvvkhl::ElementCamera>();        // Create the camera to be used
  auto optixDenoiser = std::make_shared<nvvkhl::OptixDenoiserEngine>();  // Create application elements

  app->addElement(g_elemCamera);
  app->addElement(g_elemBenchmark);
  app->addElement(optixDenoiser);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  std::vector<std::string> default_search_paths = {".", "..", "../..", "../../.."};

  // Load scene
  std::string scn_file = nvh::findFile(R"(media/cornellBox.gltf)", default_search_paths, true);
  optixDenoiser->onFileDrop(scn_file.c_str());

  // Load HDR
  std::string hdr_file = nvh::findFile(R"(media/spruit_sunrise_1k.hdr)", default_search_paths, true);
  optixDenoiser->onFileDrop(hdr_file.c_str());

  // Run as fast as possible
  app->setVsync(false);

  app->run();
  optixDenoiser.reset();
  app.reset();

  return 0;
}
