# NRC library — all host-side code that manages NRC GPU resources.
#
# Linked into GLTF_renderer by the root CMakeLists.txt.

add_library(nrc STATIC
    nrc_pass.cpp
    nrc_pass_integration.cpp
    nrc_mlp_layout.cpp
)

message(STATUS "Cmake Binary Dir: ${CMAKE_BINARY_DIR}")

target_include_directories(nrc
    PUBLIC  ${CMAKE_SOURCE_DIR}/src
    PRIVATE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
)

target_link_libraries(nrc
    PUBLIC  Vulkan::Vulkan
)
if(TARGET volk)
    target_link_libraries(nrc PUBLIC volk)
endif()

target_compile_definitions(nrc PUBLIC VK_NO_PROTOTYPES)

target_compile_features(nrc PUBLIC cxx_std_20)

# Expose Slang shader include paths via a generated .cpp so shaders that
# #include RTXNS modules can resolve them through the CMake-configured path.
set(NRC_SHADER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    "${RTXNS_ROOT}/src/NeuralShading_Shaders"
    CACHE INTERNAL "Slang shader include directories for NRC"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nrc_slang_paths.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/nrc_slang_paths.cpp
    @ONLY
)
target_sources(nrc PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/nrc_slang_paths.cpp)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Host-side test that NRC record structs match the expected byte sizes
# (cross-checked against shaders/nrc_common.h.slang in Task 6).
add_executable(nrc_feature_pack_test tests/test_feature_pack_sizes.cpp)
set_target_properties(nrc_feature_pack_test PROPERTIES FOLDER "Tests/NRC")

# Compile-only smoke test that nrc_pass.h + nrc_pass_integration.h parse.
add_executable(nrc_pass_compiles_test tests/test_nrc_pass_compiles.cpp)
target_link_libraries(nrc_pass_compiles_test PRIVATE nrc)
set_target_properties(nrc_pass_compiles_test PROPERTIES FOLDER "Tests/NRC")

# Slang shader compile test — verifies nrc_mlp_rtxns.slang facade typechecks
# against the vendored RTXNS Slang modules. Run as:
#   cmake --build . --target nrc_shader_compile_test
find_program(SLANGC slangc
    HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin
    DOC "Slang compiler (usually shipped with Vulkan SDK)"
)
if(SLANGC)
    message(STATUS "Found slangc: ${SLANGC}")
    add_custom_target(nrc_shader_compile_test
        COMMAND ${SLANGC}
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tests/test_mlp_facade.comp.slang
                -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                -I ${RTXNS_ROOT}/src/NeuralShading_Shaders
                -target spirv -profile spirv_1_5 -stage compute -entry main
                -o ${CMAKE_CURRENT_BINARY_DIR}/test_mlp_facade.spv
        COMMENT "slangc: compiling nrc_mlp_rtxns.slang facade (Task 8 test)"
        VERBATIM
    )
    set_target_properties(nrc_shader_compile_test PROPERTIES FOLDER "Tests/NRC")
else()
    message(WARNING "slangc not found — shader compile tests will be disabled")
endif()
