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

#include "MaterialDefinition.h"

BEGIN_AURORA

MaterialDefaultValues::MaterialDefaultValues(const UniformBufferDefinition& propertyDefs,
    const vector<PropertyValue>& defaultProps, const vector<TextureDefinition>& defaultTxt) :
    propertyDefinitions(propertyDefs), properties(defaultProps), textures(defaultTxt)
{
    AU_ASSERT(
        defaultProps.size() == propertyDefs.size(), "Default properties do not match definition");
    for (size_t i = 0; i < defaultTxt.size(); i++)
    {
        textureNames.push_back(defaultTxt[i].name);
    }
}

END_AURORA
