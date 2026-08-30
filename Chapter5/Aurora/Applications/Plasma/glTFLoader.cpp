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

#include "Loaders.h"
#include "SceneContents.h"

#include <glm/gtx/quaternion.hpp>

glm::mat4 getNodeTransform(const tinygltf::Node& node) {
    glm::mat4 transform(1.0f);

    // use the matrix if there is one
    if(!node.matrix.empty()) {
        transform = glm::make_mat4(node.matrix.data());
        return transform;
    }

    // otherwise construct from the TRS data
    glm::vec3 translation(0.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale(1.0f);

    if(!node.translation.empty()) {
        translation = glm::vec3(static_cast<float>(node.translation[0]),
                                static_cast<float>(node.translation[1]),
                                static_cast<float>(node.translation[2]));
    }

    if(!node.rotation.empty()) {
        rotation = glm::quat(static_cast<float>(node.rotation[3]),
                             static_cast<float>(node.rotation[0]),
                             static_cast<float>(node.rotation[1]),
                             static_cast<float>(node.rotation[2]));
    }

    if(!node.scale.empty()) {
        scale = glm::vec3(static_cast<float>(node.scale[0]),
                          static_cast<float>(node.scale[1]),
                          static_cast<float>(node.scale[2]));
    }

    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 R = glm::toMat4(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);

    transform = T * R * S;

    return transform;
}

void loadCameras(tinygltf::Model& model, SceneContents& sceneContents, std::vector<int>& nodes) {
    for(int nodeIdx : nodes) {
        tinygltf::Node& node = model.nodes[nodeIdx];
        if(node.camera != -1) {
            SceneCamera sceneCamera;
            sceneCamera.name = node.name;
            // TODO: Send the parent matrix down the recursive chain so that we support the scene hierarchy
            // This will be wrong for children at the moment
            sceneCamera.viewMatrix = glm::inverse(getNodeTransform(node));
            tinygltf::Camera& camera = model.cameras[node.camera];
            if(camera.type.compare("perspective") == 0) {
                sceneCamera.cameraType = SCENE_CAMERA_TYPE_PERSPECTIVE;
                sceneCamera.perspectiveProperties.yfov = static_cast<float>(camera.perspective.yfov);
                sceneCamera.perspectiveProperties.aspectRatio = static_cast<float>(camera.perspective.aspectRatio);
                sceneCamera.perspectiveProperties.znear = static_cast<float>(camera.perspective.znear);
                sceneCamera.perspectiveProperties.zfar = static_cast<float>(camera.perspective.zfar);
            }
            else if(camera.type.compare("orthographic") == 0) {
                sceneCamera.cameraType = SCENE_CAMERA_TYPE_ORTHOGRAPHIC;
                sceneCamera.orthographicProperties.znear = static_cast<float>(camera.orthographic.zfar);
                sceneCamera.orthographicProperties.zfar = static_cast<float>(camera.orthographic.zfar);
                sceneCamera.orthographicProperties.xmag = static_cast<float>(camera.orthographic.xmag);
                sceneCamera.orthographicProperties.ymag = static_cast<float>(camera.orthographic.ymag);
            }
            sceneContents.cameras.push_back(sceneCamera);
        }
        loadCameras(model, sceneContents, node.children);
    }
}

// Loads a glTF file into the specified renderer and scene, from the specified file path.
bool loadglTFFile(Aurora::IRenderer* /* pRenderer */, Aurora::IScene* /* pScene */,
    const string& filePath, SceneContents& sceneContents)
{
    sceneContents.cameras.clear();
    
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    string warnings, errors;

    // Attempt to load the file as a binary glTF file, which is (quickly) identified from the file
    // header. If that fails, instead try to load it as an ASCII file.
    bool result = loader.LoadBinaryFromFile(&model, &errors, &warnings, filePath);
    if(!result) {
        result = loader.LoadASCIIFromFile(&model, &errors, &errors, filePath);
    }

    // glTF is supported in a limited way just to load cameras

    for(tinygltf::Scene& scene : model.scenes) {
        loadCameras(model, sceneContents, scene.nodes);
    }
    
    return true;
}

bool saveglTFFile(Aurora::IRenderer* /* pRenderer */, Aurora::IScene* /* pScene */,
                  const string& filePath, SceneContents& sceneContents)
{
    tinygltf::TinyGLTF writer;
    tinygltf::Model model;
    tinygltf::Scene scene;
    string warnings, errors;

    if(sceneContents.cameras.size() < 1) {
        return false;
    }

    for(auto& camera : sceneContents.cameras) {
        tinygltf::Camera gltfCamera;
        gltfCamera.name = camera.name;
        gltfCamera.type = camera.cameraType == SCENE_CAMERA_TYPE_PERSPECTIVE ? "perspective" : "orthographic";
        gltfCamera.perspective.yfov = camera.perspectiveProperties.yfov;
        gltfCamera.perspective.aspectRatio = camera.perspectiveProperties.aspectRatio;
        gltfCamera.perspective.znear = camera.perspectiveProperties.znear;
        gltfCamera.perspective.zfar = camera.perspectiveProperties.zfar;

        size_t cameraIndex = model.cameras.size();
        model.cameras.push_back(gltfCamera);

        // Create a node for the camera
        tinygltf::Node cameraNode;
        cameraNode.name = camera.name;
        cameraNode.camera = static_cast<int>(cameraIndex);
        
        glm::mat4 nodeMatrix = glm::inverse(camera.viewMatrix);

        cameraNode.matrix.push_back(nodeMatrix[0][0]);
        cameraNode.matrix.push_back(nodeMatrix[0][1]);
        cameraNode.matrix.push_back(nodeMatrix[0][2]);
        cameraNode.matrix.push_back(nodeMatrix[0][3]);

        cameraNode.matrix.push_back(nodeMatrix[1][0]);
        cameraNode.matrix.push_back(nodeMatrix[1][1]);
        cameraNode.matrix.push_back(nodeMatrix[1][2]);
        cameraNode.matrix.push_back(nodeMatrix[1][3]);

        cameraNode.matrix.push_back(nodeMatrix[2][0]);
        cameraNode.matrix.push_back(nodeMatrix[2][1]);
        cameraNode.matrix.push_back(nodeMatrix[2][2]);
        cameraNode.matrix.push_back(nodeMatrix[2][3]);

        cameraNode.matrix.push_back(nodeMatrix[3][0]);
        cameraNode.matrix.push_back(nodeMatrix[3][1]);
        cameraNode.matrix.push_back(nodeMatrix[3][2]);
        cameraNode.matrix.push_back(nodeMatrix[3][3]);
        
        // Add the node to the model
        size_t nodeIndex = model.nodes.size();
        model.nodes.push_back(cameraNode);

        // Define a scene that contains the camera node
        scene.nodes.push_back(static_cast<int>(nodeIndex));
    }
    
    model.scenes.push_back(scene);
    model.defaultScene = 0;  // Set as default scene

    if(writer.WriteGltfSceneToFile(&model, filePath, false, false, true, false)) {
        return true;
    }
    else {
        return false;
    }
    
    return true;
}
