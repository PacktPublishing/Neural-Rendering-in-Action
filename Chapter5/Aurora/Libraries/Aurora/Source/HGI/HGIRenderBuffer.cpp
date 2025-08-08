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

#include "HGIRenderBuffer.h"
#include "pxr/imaging/hgi/blitCmds.h"
#include "pxr/imaging/hgi/blitCmdsOps.h"

using namespace pxr;

BEGIN_AURORA

HGIRenderBuffer::HGIRenderBuffer(
    HGIRenderer* pRenderer, uint32_t width, uint32_t height, ImageFormat format) :
    _pRenderer(pRenderer), _width(width), _height(height)
{
    switch(format) {
        case ImageFormat::Integer_RGBA: {
            _format = HgiFormat::HgiFormatUNorm8Vec4;
            break;
        }
        case ImageFormat::Half_RGBA: {
            _format = HgiFormat::HgiFormatFloat16Vec4;
            break;
        }
        case ImageFormat::Float_RGBA: {
            _format = HgiFormat::HgiFormatFloat32Vec4;
            break;
        }
        default: {
            // worst case default to rgba888
            _format = HgiFormat::HgiFormatUNorm8Vec4;
            assert(false);
            break;
        }
    }
    
    // Create descriptor for render buffer storage texture.
    HgiTextureDesc rtStorageTexDesc;
    rtStorageTexDesc.debugName  = "RT Storage Texture";
    rtStorageTexDesc.format     = _format;
    rtStorageTexDesc.dimensions = GfVec3i(width, height, 1);
    rtStorageTexDesc.layerCount = 1;
    rtStorageTexDesc.mipLevels  = 1;
    rtStorageTexDesc.usage      = HgiTextureUsageBitsShaderRead | HgiTextureUsageBitsShaderWrite;

    // Create storage texture itself.
    _storageTex = HgiTextureHandleWrapper::create(
        _pRenderer->hgi()->CreateTexture(rtStorageTexDesc), _pRenderer->hgi());
}

void HGIRenderBuffer::resize(uint32_t /*width*/, uint32_t /*height*/) {
    // TODO: Implement me!
}

const void* HGIRenderBuffer::data(size_t& stride, bool /*removePadding*/)
{
    size_t pixelSizeBytes = 0;
    switch(_format) {
        case HgiFormat::HgiFormatUNorm8Vec4: {
            pixelSizeBytes = 4;
            break;
        }
        case HgiFormat::HgiFormatFloat16Vec4: {
            pixelSizeBytes = 2 * 4;
            break;
        }
        case HgiFormat::HgiFormatFloat32Vec4: {
            pixelSizeBytes = 4 * 4;
            break;
        }
        default: {
            // worst case default to rgba888
            pixelSizeBytes = 4;
            assert(false);
            break;
        }
    }

    // Stride is always just width*pixel-size.  No row padding.
    stride                = _width * pixelSizeBytes;

    // Ensure CPU buffer is big enough for pixels.
    size_t dataByteSize = _width * _height * pixelSizeBytes;
    _mappedBuffer.resize(dataByteSize);

    // Setup commands to blit storage buffer contents to CPU buffer.
    HgiBlitCmdsUniquePtr blitCmds = _pRenderer->hgi()->CreateBlitCmds();
    {
        HgiTextureGpuToCpuOp copyOp;
        copyOp.gpuSourceTexture          = _storageTex->handle();
        copyOp.sourceTexelOffset         = GfVec3i(0);
        copyOp.mipLevel                  = 0;
        copyOp.cpuDestinationBuffer      = _mappedBuffer.data();
        copyOp.destinationByteOffset     = 0;
        copyOp.destinationBufferByteSize = dataByteSize;
        blitCmds->CopyTextureGpuToCpu(copyOp);
    }

    // Submit blocking commands.
    _pRenderer->hgi()->SubmitCmds(blitCmds.get(), HgiSubmitWaitTypeWaitUntilCompleted);

    // Return pointer to CPU buffer.
    return _mappedBuffer.data();
}

END_AURORA
