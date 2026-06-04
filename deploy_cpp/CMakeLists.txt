if(MSVC)
    add_compile_options(/utf-8)
endif()

cmake_minimum_required(VERSION 3.10)
project(ExcavatorMultiPlatform)

# 包含核心算法代码
set(CORE_SRC src/core/excavator_pipeline.cpp)
include_directories(src/core)
include_directories(3rdparty/rknn/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # ================== Android 编译分支 ==================
    message(STATUS "编译目标: Android JNI")

    # 添加 JNI 特有的库
    find_library(log-lib log)
    find_library(jnigraphics-lib jnigraphics)

    # 编译成名为 libexcavator_jni.so 的安卓库，包含 jni 接口和核心算法
    add_library(excavator_jni SHARED ${CORE_SRC} src/jni/excavator_jni.cpp)

    # 链接安卓特有的库 (OpenCV 和 RKNN 的链接路径由 NDK 决定)
    target_link_libraries(excavator_jni ${log-lib} ${jnigraphics-lib} rknn_api opencv_java4)

else()
    # ================== Linux 编译分支 ==================
    message(STATUS "编译目标: Linux Native")

    # 编译成标准的 Linux 动态库
    add_library(excavator_algo SHARED ${CORE_SRC})

    # 编译测试用的可执行程序
    add_executable(test_linux src/linux/main.cpp)
    target_link_libraries(test_linux excavator_algo rknn_api opencv_core opencv_videoio)
endif()