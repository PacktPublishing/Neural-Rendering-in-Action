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

#include "HGISampler.h"

#include "HGIRenderer.h"

BEGIN_AURORA

HGISampler::HGISampler(HGIRenderer*  pRenderer, const Properties& props)
{
    // Initailize with default values.
    _desc                 = pxr::HgiSamplerDesc();
    _desc.addressModeU    = pxr::HgiSamplerAddressModeRepeat;
    _desc.addressModeV    = pxr::HgiSamplerAddressModeRepeat;
    _desc.addressModeW    = pxr::HgiSamplerAddressModeRepeat;
    _desc.magFilter       = pxr::HgiSamplerFilterLinear;
    _desc.minFilter       = pxr::HgiSamplerFilterLinear;
    _desc.mipFilter       = pxr::HgiMipFilterLinear;
    _desc.borderColor     = pxr::HgiBorderColorOpaqueWhite;
    
    // Set the U address mode from properties.
    if (props.find(Names::SamplerProperties::kAddressModeU) != props.end())
        _desc.addressModeU = valueToHGIAddressMode(props.at(Names::SamplerProperties::kAddressModeU));

    // Set the V address mode from properties.
    if (props.find(Names::SamplerProperties::kAddressModeV) != props.end())
        _desc.addressModeV = valueToHGIAddressMode(props.at(Names::SamplerProperties::kAddressModeV));
    
    _sampler = pRenderer->hgi()->CreateSampler(_desc);
}

pxr::HgiSamplerAddressMode HGISampler::valueToHGIAddressMode(const PropertyValue& value)
{
    // Convert property string to HGI address mode.
    string valStr = value.asString();
    if (valStr.compare(Names::AddressModes::kWrap) == 0)
        return pxr::HgiSamplerAddressModeRepeat;
    if (valStr.compare(Names::AddressModes::kMirror) == 0)
        return pxr::HgiSamplerAddressModeMirrorRepeat;
    if (valStr.compare(Names::AddressModes::kClamp) == 0)
        return pxr::HgiSamplerAddressModeClampToEdge;
    if (valStr.compare(Names::AddressModes::kMirrorOnce) == 0)
        return pxr::HgiSamplerAddressModeMirrorClampToEdge;
    if (valStr.compare(Names::AddressModes::kBorder) == 0)
        return pxr::HgiSamplerAddressModeClampToBorderColor;

    // Fail if address mode not found.
    AU_FAIL("Unknown address mode:%s", value.asString().c_str());
    return pxr::HgiSamplerAddressModeCount;
}

END_AURORA
