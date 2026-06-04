if(MSVC)
    add_compile_options(/utf-8)
endif()

cmake_minimum_required(VERSION 3.10)
project(ExcavatorMultiPlatform)

# 包含核心算法代码
set(CORE_SRC src/core/excavator_pipeline.cpp)
include_directories(src/core)
include_directories(3rdparty/rknn/include)
include_directories(3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/include)

set(NDK_SYSROOT "C:/Users/GT15/AppData/Local/Android/Sdk/ndk/30.0.14904198/toolchains/llvm/prebuilt/windows-x86_64/sysroot")

# 引入基础的 jni.h 等标准库头文件
include_directories(${NDK_SYSROOT}/usr/include)

# 引入 Android 特定的头文件 (如 android/asset_manager.h, android/log.h)
include_directories(${NDK_SYSROOT}/usr/include/aarch64-linux-android)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # ================== Android 编译分支 ==================
    message(STATUS "编译目标: Android JNI")

    # 👇 [新增] 告诉 CMake 你的第三方 .so 库在哪个文件夹里
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/rknn/android/arm64-v8a)
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/libs/arm64-v8a)

    # 添加 JNI 特有的系统库 (日志和图形缓冲)
    find_library(log-lib log)
    find_library(jnigraphics-lib jnigraphics)
    find_library(android-lib android)

    # 编译成名为 libexcavator_jni.so 的安卓库，包含 jni 接口和核心算法
    add_library(excavator_jni SHARED ${CORE_SRC} src/jni/excavator_jni.cpp)

    # 👇 [修改] 链接库名称。RKNN 的库名通常是 rknnrt (对应 librknnrt.so)
    target_link_libraries(excavator_jni ${log-lib} ${jnigraphics-lib} ${android-lib} rknnrt opencv_java4)
else()
    # ================== Linux 编译分支 ==================
    message(STATUS "编译目标: Linux Native")

    # 编译成标准的 Linux 动态库
    add_library(excavator_algo SHARED ${CORE_SRC})

    # 编译测试用的可执行程序
    add_executable(test_linux src/linux/main.cpp)
    target_link_libraries(test_linux excavator_algo rknn_api opencv_core opencv_videoio)
endif()