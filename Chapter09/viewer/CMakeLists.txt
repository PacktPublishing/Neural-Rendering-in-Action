# Chapter 9, Part 2 -- real-time Gaussian splat viewer built on Vulkan and LightweightVK (https://github.com/corporateshark/lightweightvk)
#
# This is the book's only C++ target, so it is one single CMake project: there is no root CMakeLists any more.
# This one file bootstraps the dependencies, configures LightweightVK and builds the viewer.
#
#     cmake -S Chapter09/viewer -B .build
#     cmake --build .build --config Release
#     ./.build/Release/splat_viewer
#
# Pass -DLVK_ROOT=<path> to build against a different custom LightweightVK checkout.

cmake_minimum_required(VERSION 3.22)

project(NeuralRenderingInAction LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(BUILD_SHARED_LIBS OFF)

# The repository root, two levels above this file. Everything the viewer pulls
# in -- RPly, nlohmann/json, LightweightVK -- lives under `deps/` there.
cmake_path(SET BOOK_ROOT NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if(NOT DEFINED LVK_ROOT)
  set(LVK_ROOT "${BOOK_ROOT}/deps/src/lightweightvk")
endif()
cmake_path(SET LVK_ROOT NORMALIZE "${LVK_ROOT}")

if($ENV{CLION_IDE})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BOOK_ROOT})
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Two bootstraps, and both are needed: ours fetches LightweightVK, RPly,
# nlohmann/json and the pre-trained scenes; LightweightVK's own fetches the
# tree it vendors nothing of -- GLFW, glm, ImGui, glslang, stb.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(ENV{LVK_WITH_SLANG} "OFF")

set(DEPS_CACHE_FILE "${BOOK_ROOT}/deps/.bootstrap.json")

# Ours lives beside this directory, in Chapter09, and resolves `deps/` itself;
# LightweightVK's is its own script, run from its own root.
cmake_path(SET DEPLOY_DEPS NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../deploy_deps.py")

execute_process(COMMAND ${Python3_EXECUTABLE} "${DEPLOY_DEPS}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${Python3_EXECUTABLE} deploy_deps.py
                WORKING_DIRECTORY ${LVK_ROOT} COMMAND_ERROR_IS_FATAL ANY)

add_custom_command(COMMAND ${Python3_EXECUTABLE} "${DEPLOY_DEPS}" OUTPUT ${DEPS_CACHE_FILE}
                   DEPENDS "${BOOK_ROOT}/deps/bootstrap.json")
add_custom_target(Dependencies DEPENDS ${DEPS_CACHE_FILE})
set_property(TARGET Dependencies PROPERTY FOLDER "third-party")

if(NOT EXISTS ${DEPS_CACHE_FILE})
  message(FATAL_ERROR "Cannot bootstrap dependencies.")
endif()
if(NOT EXISTS "${LVK_ROOT}/lvk/LVK.h")
  message(FATAL_ERROR "LightweightVK not found at ${LVK_ROOT}. Run 'python Chapter09/deploy_deps.py', "
                      "or re-run CMake with -DLVK_ROOT=<path>.")
endif()
foreach(dep "deps/src/rply/rply.c" "deps/src/json/single_include/nlohmann/json.hpp")
  if(NOT EXISTS "${BOOK_ROOT}/${dep}")
    message(FATAL_ERROR "${dep} not found. Run 'python Chapter09/deploy_deps.py'.")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# LightweightVK
# ---------------------------------------------------------------------------
# Set without FORCE, so any of these can still be overridden from the command
# line. Added before the definitions below, which are meant for our own target
# and not for LightweightVK's.

set(LVK_WITH_SAMPLES   OFF CACHE BOOL "")
set(LVK_WITH_TRACY     OFF CACHE BOOL "")
set(LVK_WITH_TRACY_GPU OFF CACHE BOOL "")
set(LVK_WITH_SLANG     OFF CACHE BOOL "")
set(LVK_WITH_GLFW      ON  CACHE BOOL "")
set(LVK_DEPLOY_SCREENSHOT_TESTS OFF CACHE BOOL "")

message(STATUS "Chapter 9 viewer: LightweightVK from ${LVK_ROOT}")
add_subdirectory("${LVK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/lightweightvk")

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------
# One translation unit of our own, plus RPly's and LightweightVK's sample
# harness. Everything else -- the camera, the maths, the PNG writer, the
# windowing, the main loop -- comes from the dependency tree.

add_executable(splat_viewer
  main.cpp
  "${BOOK_ROOT}/deps/src/rply/rply.c"
  "${LVK_ROOT}/samples/VulkanApp.cpp")

# RPly passes an uninitialized `ply` to the error callback on the type-check
# failure path -- rply.c:379 and :481 both read it before `ply_alloc()` -- and
# MSVC is right to say so (C4700). Debug builds report it; release ones do not,
# because the optimizer removes the read. The path only fires on a platform
# whose type sizes RPly cannot represent, and patching a vendored file to
# silence a warning is worse than the warning, so suppress it for this one
# source file rather than for the whole target.
if(MSVC)
  set_source_files_properties("${BOOK_ROOT}/deps/src/rply/rply.c"
                              PROPERTIES COMPILE_OPTIONS "/wd4700")
endif()

set_target_properties(splat_viewer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(COMMAND lvk_set_folder)
  lvk_set_folder(splat_viewer "Chapter09")
endif()
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT splat_viewer)

target_include_directories(splat_viewer PRIVATE
  "${BOOK_ROOT}/deps/src/rply"                                       # rply.h
  "${BOOK_ROOT}/deps/src/json/single_include"                        # nlohmann/json.hpp
  "${LVK_ROOT}/samples"                                              # VulkanApp.h
  "${LVK_ROOT}/third-party/deps/src"                                 # stb/stb_image.h
  "${LVK_ROOT}/third-party/deps/src/glm"                             # glm
  "${LVK_ROOT}/third-party/deps/src/3D-Graphics-Rendering-Cookbook")  # shared/Camera.h
# LVKstb carries the stb implementations VulkanApp's screenshot writer needs.
target_link_libraries(splat_viewer PRIVATE LVKLibrary glfw LVKstb)

if(MSVC)
  target_compile_definitions(splat_viewer PRIVATE _CONSOLE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()
if(WIN32)
  target_compile_definitions(splat_viewer PRIVATE VK_USE_PLATFORM_WIN32_KHR=1)
endif()
