// Copyright 2025 Autodesk, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "pch.h"

#include "HGIImage.h"
#include "HGIRenderer.h"

using namespace pxr;

BEGIN_AURORA

HGIImage::HGIImage(HGIRenderer* pRenderer, const IImage::InitData& initData)
{
    // Convert Aurora image format to HGI image format.
    uint32_t pixelSizeBytes;
    HgiFormat hgiFormat = getHGIFormat(initData.format, initData.linearize, &pixelSizeBytes);

    // Create descriptor for image.
    HgiTextureDesc imageTexDesc;
    imageTexDesc.debugName      = initData.name;
    imageTexDesc.format         = hgiFormat;
    imageTexDesc.dimensions     = GfVec3i(initData.width, initData.height, 1);
    imageTexDesc.layerCount     = 1;
    imageTexDesc.mipLevels      = 1;
    imageTexDesc.usage          = HgiTextureUsageBitsShaderRead;
    imageTexDesc.pixelsByteSize = static_cast<size_t>(initData.width)
        * static_cast<size_t>(initData.height) * static_cast<size_t>(pixelSizeBytes);
    imageTexDesc.initialData    = initData.pImageData;

    // Create the texture.
    _texture = HgiTextureHandleWrapper::create(
        pRenderer->hgi()->CreateTexture(imageTexDesc), pRenderer->hgi());
    
    if(initData.isEnvironment) {
        // Create Alias Map buffer object.
        size_t width = initData.width;
        size_t height = initData.height;
        
        HgiBufferDesc aliasMapDataUboDesc;
        aliasMapDataUboDesc.debugName = "Raytracing alias map global data UBO";
        aliasMapDataUboDesc.usage     = HgiBufferUsageUniform;
        aliasMapDataUboDesc.byteSize  = sizeof(AliasMap::Entry) * width * height;
        _pAliasMapBuffer = HgiBufferHandleWrapper::create(pRenderer->hgi()->CreateBuffer(aliasMapDataUboDesc), pRenderer->hgi());
        
        // Create per instance data buffer
        AliasMap::Entry* aliasMapDta = new AliasMap::Entry[width * height];
                    
        // Ensure CPU buffer is big enough for pixels.
        size_t dataByteSize = width * height * 32;
        std::vector<uint8_t> _mappedBuffer;
        _mappedBuffer.resize(dataByteSize);
        
        // Setup commands to blit storage buffer contents to CPU buffer.
        HgiBlitCmdsUniquePtr blitCmdsEnvMap = pRenderer->hgi()->CreateBlitCmds();
        {
            HgiTextureGpuToCpuOp copyOpEnvMap;
            copyOpEnvMap.gpuSourceTexture          = texture();
            copyOpEnvMap.sourceTexelOffset         = GfVec3i(0);
            copyOpEnvMap.mipLevel                  = 0;
            copyOpEnvMap.cpuDestinationBuffer      = _mappedBuffer.data();
            copyOpEnvMap.destinationByteOffset     = 0;
            copyOpEnvMap.destinationBufferByteSize = dataByteSize;
            blitCmdsEnvMap->CopyTextureGpuToCpu(copyOpEnvMap);
        }
        pRenderer->hgi()->SubmitCmds(blitCmdsEnvMap.get(), HgiSubmitWaitTypeWaitUntilCompleted);
        
        AliasMap::build((const float*)_mappedBuffer.data(), uvec2(width, height), aliasMapDta, sizeof(AliasMap::Entry) * width * height, _luminanceIntegral);

        pxr::HgiBlitCmdsUniquePtr blitCmdsAliasMap = pRenderer->hgi()->CreateBlitCmds();
        pxr::HgiBufferCpuToGpuOp blitOpAliasMap;
        blitOpAliasMap.byteSize              = sizeof(AliasMap::Entry)  * width * height;
        blitOpAliasMap.cpuSourceBuffer       = aliasMapDta;
        blitOpAliasMap.sourceByteOffset      = 0;
        blitOpAliasMap.gpuDestinationBuffer  = aliasMap();
        blitOpAliasMap.destinationByteOffset = 0;
        blitCmdsAliasMap->CopyBufferCpuToGpu(blitOpAliasMap);
        pRenderer->hgi()->SubmitCmds(blitCmdsAliasMap.get());
    }
}

HgiFormat HGIImage::getHGIFormat(ImageFormat format, bool linearize, uint32_t* pPixelByteSizeOut)
{
    // Convert the HGI format to HGI format enum.
    switch (format)
    {
    case ImageFormat::Integer_RGBA:
        *pPixelByteSizeOut = 4;
        return linearize ? HgiFormat::HgiFormatUNorm8Vec4srgb : HgiFormat::HgiFormatUNorm8Vec4;
    case ImageFormat::Float_RGBA:
        *pPixelByteSizeOut = 16;
        return HgiFormat::HgiFormatFloat32Vec4;
    case ImageFormat::Float_RGB:
        *pPixelByteSizeOut = 12;
        return HgiFormat::HgiFormatFloat32Vec3;
    default:
        break;
    }

    AU_FAIL("Unsupported image format:%x", format);
    return (HgiFormat)-1;
}

END_AURORA
