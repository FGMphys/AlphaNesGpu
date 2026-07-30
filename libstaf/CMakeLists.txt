cmake_minimum_required(VERSION 3.16)
project(libstaf LANGUAGES C CXX)

option(STAF_WITH_ORT "Link ONNX Runtime (optional)" OFF)
option(STAF_WITH_JMD "Link neuralmdGPU CUDA AF/force + nn_nn_mlp" ON)
option(STAF_BUILD_SMOKE "Build smoke binaries" ON)

# Override on a new machine: -DORT_ROOT=... -DCUDA_HOME=... -DCUDNN_LIB=...
# Runtime discovery for LD_LIBRARY_PATH: source scripts/staf_gpu_env.sh
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-cuda11" CACHE PATH
    "ONNX Runtime root (cuda11 build on this box; cuda12 tarball on newer hosts)")
set(JMD_SRC "/home/francegm/neuralmdGPU/full_atom/src" CACHE PATH "neuralmdGPU src")
set(CUDA_HOME "/home/francegm/programmi/cuda" CACHE PATH "CUDA toolkit (11.8 here; 12.x elsewhere)")
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
  add_library(staf_jmd_nn STATIC
    vendor/jmd/nn_nn_mlp.cu
    vendor/jmd/src_nn/descriptor_builder/reforce.cu
    vendor/jmd/src_nn/staf_force_pbc.cu
    vendor/jmd/src_nn/force/rad/reforce.cu
    vendor/jmd/src_nn/force/ang/reforce.cu
  )
  target_include_directories(staf_jmd_nn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/jmd
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JMD_SRC}
  )
  set_target_properties(staf_jmd_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_compile_options(staf_jmd_nn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=${CUDA_ARCH} --use_fast_math>
  )

  # Prebuilt CUDA/C objects from neuralmdGPU (exclude nn_nn.o / main.o / md.o)
  set(JMD_OBJS
    ${JMD_SRC}/celle_gpu.o
    ${JMD_SRC}/nn_io.o
    ${JMD_SRC}/nn_smart_allocator.o
    ${JMD_SRC}/nn_smart_allocator_gpu.o
    ${JMD_SRC}/secure_search.o
    ${JMD_SRC}/io.o
    ${JMD_SRC}/log.o
    ${JMD_SRC}/vector.o
    ${JMD_SRC}/interaction_map.o
    ${JMD_SRC}/src_nn/fingerprint/rad/reforce.o
    ${JMD_SRC}/src_nn/fingerprint/ang/reforce.o
  )
  foreach(o ${JMD_OBJS})
    if(NOT EXISTS ${o})
      message(FATAL_ERROR "Missing JMD object: ${o} — build neuralmdGPU full_atom first")
    endif()
  endforeach()

  target_include_directories(staf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/jmd
    ${JMD_SRC}
    ${CUDA_HOME}/include
  )
  target_link_libraries(staf PUBLIC staf_jmd_nn ${JMD_OBJS}
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
    add_executable(staf_virial_batch src/runtime/staf_virial_batch.cpp)
    target_link_libraries(staf_virial_batch PRIVATE staf)
  endif()
endif()

install(TARGETS staf ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
