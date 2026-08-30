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
#pragma once

#include <pxr/imaging/hgi/sampler.h>

BEGIN_AURORA

// Forward declarations.
class HGIRenderer;

// An internal implementation for ISampler.
class HGISampler : public ISampler
{
public:
    /*** Lifetime Management ***/

    HGISampler(HGIRenderer* pRenderer, const Properties& props);
    ~HGISampler() {};
    
    
    const pxr::HgiSamplerHandle sampler() const { return _sampler; }
    const pxr::HgiSamplerDesc* desc() const { return &_desc; }

private:
    static pxr::HgiSamplerAddressMode valueToHGIAddressMode(const PropertyValue& value);
        
    pxr::HgiSamplerDesc _desc;
    pxr::HgiSamplerHandle _sampler;
};

MAKE_AURORA_PTR(HGISampler);

END_AURORA
