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

#ifndef PATHTRACINGCOMMON_H
#define PATHTRACINGCOMMON_H

// Define this symbol for NRD.
//#define COMPILER_DXC

#include "BSDFCommon.metal"
#include "Colors.metal"
#include "Environment.metal"
#include "Frame.metal"
#include "Geometry.metal"
#include "Globals.metal"
#include "GroundPlane.metal"
#include "Material.metal"
#include "Random.metal"
#include "RayTrace.metal"
#include "Sampling.metal"
#include "GlobalPipelineState.metal"

// Collects the full set of property values for an environment.
Environment prepareEnvironmentValues()
{
    Environment values;
    values.constants         = gEnvironmentConstants;
    values.backgroundTexture = gEnvironmentBackgroundTexture;
#if DIRECTX
    values.sampler           = gSamplerArray[0];
#else
    values.sampler           = gDefaultSampler;
#endif
    values.lightTexture      = gEnvironmentLightTexture;
    return values;
}

// Adjusts the specified radiance, to clamp extreme values and detect errors.
void adjustRadiance(float maxLuminance, bool displayErrors, thread float3& radiance)
{
    // Clamp result colors above a certain luminance threshold, to minimize fireflies.
    // NOTE: This biases the final result and should be used carefully.
    float lum = luminance(radiance);
    if (lum > maxLuminance)
    {
        radiance *= maxLuminance / lum;
    }

    // Replace an invalid radiance sample with a diagnostic (infinite) value when displaying errors,
    // or black otherwise. Shading errors are usually the result of bad geometry (e.g. zero-length
    // normals), but may also be caused by internal shading errors that should be addressed.
    const float3 NAN_COLOR = float3(INFINITY, 0.0f, 0.0f); // red
    const float3 INF_COLOR = float3(0.0f, INFINITY, 0.0f); // green
    if (any(isnan(radiance)))
    {
        radiance = displayErrors ? NAN_COLOR : BLACK;
    }
    else if (any(isinf(radiance)))
    {
        radiance = displayErrors ? INF_COLOR : BLACK;
    }
}

#endif // !PATHTRACINGCOMMON_H
