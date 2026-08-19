cmake_minimum_required(VERSION 3.16)
project(libstaf_cg LANGUAGES C CXX)

option(STAF_WITH_ORT "Link ONNX Runtime (optional)" OFF)
option(STAF_WITH_JMD "Link CG CUDA AF/force + nn_nn_mlp" ON)
option(STAF_BUILD_SMOKE "Build smoke binaries" ON)

# Same discovery knobs as libstaf/CMakeLists.txt (ORT_ROOT / CUDA_HOME / CUDNN_LIB).
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-cuda11" CACHE PATH
    "ONNX Runtime root (cuda11 build on this box; cuda12 tarball on newer hosts)")
set(CUDA_HOME "/home/francegm/programmi/cuda" CACHE PATH "CUDA toolkit")
set(CUDNN_LIB "/home/francegm/programmi/cudaNN/cuda-11.8/lib" CACHE PATH "cuDNN lib dir")
set(CUDA_ARCH "sm_70" CACHE STRING "nvcc arch")

add_library(staf
  src/staf_api.cpp
  src/mlp/staf_mlp_ort.cc
)
target_include_directories(staf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(staf PUBLIC cxx_std_14)
target_compile_options(staf PRIVATE -Wall -O2)
target_compile_definitions(staf PUBLIC STAF_WITH_ORT=$<BOOL:${STAF_WITH_ORT}> STAF_WITH_JMD=$<BOOL:${STAF_WITH_JMD}>)

if(STAF_WITH_ORT)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ORT_ROOT}/include)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ORT_ROOT}/lib)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ORT not found under ${ORT_ROOT}")
  endif()
  target_include_directories(staf PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(staf PUBLIC ${ONNXRUNTIME_LIBRARY})
  target_link_options(staf PUBLIC
    -Wl,-rpath,${ORT_ROOT}/lib
    -Wl,-rpath,${CUDNN_LIB}
    -Wl,-rpath,${CUDA_HOME}/lib64
  )
  target_link_directories(staf PUBLIC ${CUDNN_LIB} ${CUDA_HOME}/lib64)
endif()

if(STAF_WITH_JMD)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  # Compile CG sources here (no prebuilt neuralmdGPU .o / JMD_OBJS).
  add_library(staf_jmd_nn STATIC
    vendor/jmd/nn_nn_mlp.cu
    vendor/jmd/celle_gpu.cu
    vendor/jmd/nn_smart_allocator_gpu.cu
    vendor/jmd/src_nn/descriptor_builder/reforce.cu
    vendor/jmd/src_nn/fingerprint/rad/reforce.cu
    vendor/jmd/src_nn/fingerprint/ang/reforce.cu
    vendor/jmd/src_nn/force/rad/reforce.cu
    vendor/jmd/src_nn/force/ang/reforce.cu
    vendor/jmd/vector.c
    vendor/jmd/io.c
    vendor/jmd/log.c
    vendor/jmd/interaction_map.c
    vendor/jmd/nn_io.c
    vendor/jmd/nn_smart_allocator.c
    vendor/jmd/secure_search.c
  )
  target_include_directories(staf_jmd_nn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/jmd
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  set_target_properties(staf_jmd_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(staf_jmd_nn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=${CUDA_ARCH} --use_fast_math>
    $<$<COMPILE_LANGUAGE:C>:-O2 -Wall>
  )

  target_include_directories(staf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/jmd
    ${CUDA_HOME}/include
  )
  target_link_libraries(staf PUBLIC staf_jmd_nn
    cudart cuda
    gsl gslcblas m dl pthread
  )
  target_link_directories(staf PUBLIC ${CUDA_HOME}/lib64 /usr/lib/x86_64-linux-gnu)
  target_link_options(staf PUBLIC
    -Wl,-rpath,${CUDA_HOME}/lib64
  )
endif()

if(STAF_BUILD_SMOKE)
  add_executable(staf_mlp_smoke src/mlp/staf_mlp_smoke.cc)
  target_link_libraries(staf_mlp_smoke PRIVATE staf)

  if(STAF_WITH_JMD)
    add_executable(staf_force_smoke src/runtime/staf_force_smoke.cpp)
    target_link_libraries(staf_force_smoke PRIVATE staf)
  endif()
endif()

install(TARGETS staf ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
