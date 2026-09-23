# NRC library — all host-side code that manages NRC GPU resources.
#
# Linked into GLTF_renderer by the root CMakeLists.txt.

add_library(nrc STATIC
    nrc_pass.cpp
    nrc_pass_integration.cpp
    nrc_mlp_layout.cpp
)

set(NRC_SLANG_LIBRARY_FILES
    shaders/nrc.slang
    shaders/nrc/records.slang
    shaders/nrc/encoders.slang
    shaders/nrc/features.slang
    shaders/nrc/estimator.slang
    shaders/nrc/network.slang
    shaders/nrc_renderer_adapter.h.slang
)
set_source_files_properties(${NRC_SLANG_LIBRARY_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(nrc PRIVATE ${NRC_SLANG_LIBRARY_FILES})
source_group("Shaders/Modules" FILES ${NRC_SLANG_LIBRARY_FILES})

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

# Expose Slang module search paths via a generated .cpp so the runtime compiler
# can resolve NRC and RTXNS imports through the CMake-configured paths.
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

# Host-side smoke test for the frozen record allocation sizes. Compiled Slang
# layout compatibility is checked during refactor verification.
add_executable(nrc_feature_pack_test tests/test_feature_pack_sizes.cpp)
set_target_properties(nrc_feature_pack_test PROPERTIES FOLDER "Tests/NRC")

# Compile-only smoke test that nrc_pass.h + nrc_pass_integration.h parse.
add_executable(nrc_pass_compiles_test tests/test_nrc_pass_compiles.cpp)
target_link_libraries(nrc_pass_compiles_test PRIVATE nrc)
set_target_properties(nrc_pass_compiles_test PROPERTIES FOLDER "Tests/NRC")

# Slang shader compile test — verifies that the multi-file nrc module
# typechecks against RTXNS with the same compiler used by production shaders.
# Run as:
#   cmake --build . --target nrc_shader_compile_test
if(Slang_SLANGC_EXECUTABLE)
    message(STATUS "NRC shader tests use slangc: ${Slang_SLANGC_EXECUTABLE}")
    add_custom_target(nrc_shader_compile_test
        COMMAND ${Slang_SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tests/test_mlp_facade.comp.slang
                -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                -I ${RTXNS_ROOT}/src/NeuralShading_Shaders
                -target spirv -profile spirv_1_6
                -capability spvCooperativeVectorNV+spvCooperativeVectorTrainingNV
                -restrictive-capability-check
                -emit-spirv-directly
                -matrix-layout-row-major
                -force-glsl-scalar-layout
                -stage compute -entry main
                -o ${CMAKE_CURRENT_BINARY_DIR}/test_mlp_facade.spv
        COMMENT "slangc: compiling the NRC Slang module"
        VERBATIM
    )
    set_target_properties(nrc_shader_compile_test PROPERTIES FOLDER "Tests/NRC")

    # Execute the real estimator helpers on a Vulkan compute device. This
    # shader does not instantiate the network and needs no CoopVec extension.
    set(NRC_TRANSPORT_TEST_SPV "${CMAKE_CURRENT_BINARY_DIR}/test_nrc_transport.spv")
    add_custom_command(
        OUTPUT ${NRC_TRANSPORT_TEST_SPV}
        COMMAND ${Slang_SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nrc_transport.comp.slang
                -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                -I ${RTXNS_ROOT}/src/NeuralShading_Shaders
                -target spirv -profile spirv_1_6
                -restrictive-capability-check -emit-spirv-directly
                -matrix-layout-row-major -force-glsl-scalar-layout
                -stage compute -entry main
                -o ${NRC_TRANSPORT_TEST_SPV}
        DEPENDS tests/test_nrc_transport.comp.slang ${NRC_SLANG_LIBRARY_FILES}
        COMMENT "slangc: compiling deterministic NRC transport fixtures"
        VERBATIM
    )
    add_executable(nrc_transport_test tests/test_nrc_transport.cpp ${NRC_TRANSPORT_TEST_SPV})
    set_source_files_properties(${NRC_TRANSPORT_TEST_SPV} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_link_libraries(nrc_transport_test PRIVATE Vulkan::Vulkan)
    target_compile_features(nrc_transport_test PRIVATE cxx_std_17)
    set_target_properties(nrc_transport_test PROPERTIES FOLDER "Tests/NRC")
    add_custom_target(nrc_transport_check
        COMMAND $<TARGET_FILE:nrc_transport_test> ${NRC_TRANSPORT_TEST_SPV}
        DEPENDS nrc_transport_test
        COMMENT "Executing production NRC transport helpers on Vulkan"
        VERBATIM
    )
    set_target_properties(nrc_transport_check PROPERTIES FOLDER "Tests/NRC")

    # Check populated-pool coverage and balanced repetition with the exact
    # production GPU sampler, including arbitrary non-power-of-two counts.
    set(NRC_SAMPLER_TEST_SPV "${CMAKE_CURRENT_BINARY_DIR}/test_nrc_sampler.spv")
    add_custom_command(
        OUTPUT ${NRC_SAMPLER_TEST_SPV}
        COMMAND ${Slang_SLANGC_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nrc_sampler.comp.slang
                -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                -I ${RTXNS_ROOT}/src/NeuralShading_Shaders
                -target spirv -profile spirv_1_6
                -restrictive-capability-check -emit-spirv-directly
                -matrix-layout-row-major -force-glsl-scalar-layout
                -stage compute -entry main
                -o ${NRC_SAMPLER_TEST_SPV}
        DEPENDS tests/test_nrc_sampler.comp.slang ${NRC_SLANG_LIBRARY_FILES}
        COMMENT "slangc: compiling NRC minibatch coverage regression"
        VERBATIM
    )
    add_executable(nrc_sampler_test tests/test_nrc_sampler.cpp ${NRC_SAMPLER_TEST_SPV})
    set_source_files_properties(${NRC_SAMPLER_TEST_SPV} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_link_libraries(nrc_sampler_test PRIVATE Vulkan::Vulkan)
    target_compile_features(nrc_sampler_test PRIVATE cxx_std_17)
    set_target_properties(nrc_sampler_test PROPERTIES FOLDER "Tests/NRC")
    add_custom_target(nrc_sampler_check
        COMMAND $<TARGET_FILE:nrc_sampler_test> ${NRC_SAMPLER_TEST_SPV}
        DEPENDS nrc_sampler_test
        COMMENT "Executing production NRC minibatch sampler on Vulkan"
        VERBATIM
    )
    set_target_properties(nrc_sampler_check PROPERTIES FOLDER "Tests/NRC")
else()
    message(WARNING "Slang package has no slangc executable — NRC shader compile test disabled")
endif()

# Run the production mixed-precision training and optimizer shaders on analytic
# network fixtures. Requires VK_NV_cooperative_vector with FP32 accumulation.
if(Slang_SLANGC_EXECUTABLE)
    set(NRC_PRECISION_TEST_SPVS)
    foreach(pass train optimizer record_complete)
        set(spv "${CMAKE_CURRENT_BINARY_DIR}/test_nrc_${pass}.spv")
        add_custom_command(
            OUTPUT ${spv}
            COMMAND ${Slang_SLANGC_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/nrc_${pass}.comp.slang
                    -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                    -I ${RTXNS_ROOT}/src/NeuralShading_Shaders
                    -target spirv -profile spirv_1_6
                    -capability spvCooperativeVectorNV+spvCooperativeVectorTrainingNV
                    -restrictive-capability-check -emit-spirv-directly
                    -matrix-layout-row-major -force-glsl-scalar-layout
                    -stage compute -entry main -o ${spv}
            DEPENDS shaders/nrc_${pass}.comp.slang ${NRC_SLANG_LIBRARY_FILES}
            COMMENT "slangc: compiling production NRC ${pass} precision fixture"
            VERBATIM
        )
        list(APPEND NRC_PRECISION_TEST_SPVS ${spv})
    endforeach()
    add_executable(nrc_precision_test tests/test_nrc_precision.cpp ${NRC_PRECISION_TEST_SPVS})
    set_source_files_properties(${NRC_PRECISION_TEST_SPVS} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_link_libraries(nrc_precision_test PRIVATE Vulkan::Vulkan)
    target_compile_features(nrc_precision_test PRIVATE cxx_std_17)
    set_target_properties(nrc_precision_test PROPERTIES FOLDER "Tests/NRC")
    add_custom_target(nrc_precision_check
        COMMAND $<TARGET_FILE:nrc_precision_test> ${NRC_PRECISION_TEST_SPVS}
        DEPENDS nrc_precision_test
        COMMENT "Executing production NRC training and Adam precision regressions"
        VERBATIM
    )
    set_target_properties(nrc_precision_check PROPERTIES FOLDER "Tests/NRC")
endif()
