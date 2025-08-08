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

// A class representing a virtual camera that can be updated with mouse / keyboard input, and
// supplies the corresponding view and projection matrices.
class Camera
{
public:
    /*** Types ***/

    struct Inputs
    {
        bool LeftButton;
        bool MiddleButton;
        bool RightButton;
        bool Wheel;
    };

    /*** Functions ***/

    const vec3& eye() const { return _eye; }
    const vec3& target() const { return _target; }
    const vec3& upDir() const { return _up; }
    vec3 forwardDir() const { return normalize(_target - _eye); }
    vec3 rightDir() const { return cross(forwardDir(), _up); }
    float targetDistance() const { return length(_target - _eye); }
    float aspectRatio() const { return _aspectRatio; }
    const mat4& viewMatrix();
    const mat4& projMatrix();
    const ivec2& dimensions() const { return _dimensions; }
    bool isDirty() { return _isViewDirty || _isProjDirty; }
    void setIsOrtho(bool value);
    void setView(const vec3& eye, const vec3& target);
    void setProjection(float fov, float nearClip, float farClip);
    void setAspect(float aspect);
    void setDimensions(const ivec2& dimensions);
    void fit(const Foundation::BoundingBox& bounds);
    void fit(const Foundation::BoundingBox& bounds, const vec3& direction);
    void mouseMove(int xPos, int yPos, const Inputs& inputs);
    float fov() const { return _fov; }
    float getNear() const { return _near; }
    float getFar() const { return _far; }
    bool isOrtho() const { return _isOrtho; }
    
    void moveTo(const mat4& destViewMatrix);
    void update(float delta);
    bool isMoving() const { return (_moveT < 1.0f); }
    
private:
    /*** Private Functions ***/
    
    void orbit(const vec2& delta);
    void pan(const vec2& delta);
    void dolly(const vec2& delta);
    
    /*** Private Variables ***/
    
    bool _isUpdating = false;
    vec2 _lastMouse;
    bool _isViewDirty = true;
    bool _isProjDirty = true;
    bool _isOrtho     = false;
    float _fov        = radians(45.0f);
    float _near       = 0.1f;
    float _far        = 1.0f;
    float _azimuth    = 0.0f;
    float _elevation  = 0.0f;
    vec3 _eye         = vec3(0.0f, 0.0f, 1.0f);
    vec3 _target      = vec3(0.0f, 0.0f, 0.0f);
    vec3 _up          = vec3(0.0f, 1.0f, 0.0f);
    ivec2 _dimensions = ivec2(100, 100);
    float _aspectRatio = 1.0f;
    mat4 _viewMatrix;
    mat4 _projMatrix;
    
    float _moveT = 1.0f;
    mat4 _destView, _fromView;
    float _destFov, _destNear, _destFar, _destAspect;
    float _fromFov, _fromNear, _fromFar, _fromAspect;


    mat4 tweenCameraViewMatrix(const mat4& from, const mat4& to, float t);
    float easeInOutCubic(float t);

};
