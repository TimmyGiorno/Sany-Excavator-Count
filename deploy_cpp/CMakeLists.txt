cmake_minimum_required(VERSION 3.10)
project(SanyExcavatorCountingAlg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine whether Visual Studio Compiler
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Select one Backend
option(USE_RKNN  "Rockchip NPU backend (Android / Linux ARM)"  OFF)
option(USE_ONNX  "ONNX Runtime backend (Windows / Linux x86)"  OFF)
option(USE_MOCK  "Mock backend for pipeline benchmarking"      ON )

if(USE_RKNN)
    add_definitions(-DUSE_RKNN)
    set(BACKEND_SRC src/backends/rknn_backend.cpp)
    set(BACKEND_INCLUDE jni)
    message(STATUS "Backend: RKNN")
elseif(USE_ONNX)
    add_definitions(-DUSE_ONNX)
    set(BACKEND_SRC src/backends/onnx_backend.cpp)
    message(STATUS "Backend: ONNX Runtime")
else()
    add_definitions(-DUSE_MOCK)
    set(BACKEND_SRC src/backends/mock_backend.cpp)
    message(STATUS "Backend: Mock (benchmarking)")
endif()

# Sources
set(COMMON_SRC
    src/inference_engine.cpp
    src/excavator_tracker.cpp
)

set(ENGINE_SRC
    ${COMMON_SRC}
    ${BACKEND_SRC}
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/${BACKEND_INCLUDE}
)


# Target: excavator_algo  (shared library, Android via CMake)
add_library(excavator_algo SHARED
        ${ENGINE_SRC}
        jni/excavator_jni.cpp)
target_compile_options(excavator_algo PRIVATE -Wall -O2 -fPIC -fexceptions -frtti)
target_link_libraries(excavator_algo rknnrt log android)


# Target: test_video  (Windows desktop performance test)
#    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
#
#    add_executable(test_video
#        src/main.cpp
#        ${ENGINE_SRC}
#    )
#
#    set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv/windows_x64/build/x64/vc16/lib")
#    find_package(OpenCV 4 REQUIRED)
#    set(ORT_DIR "${THIRD_PARTY_DIR}/onnxruntime/windows_x64")
#    include_directories(${ORT_DIR}/include)
#    set(ONNXRUNTIME_LIB "${ORT_DIR}/lib/onnxruntime.lib")
#    target_link_libraries(test_video ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
