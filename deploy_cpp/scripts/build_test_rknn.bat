@echo off

:: 强制切换控制台编码为 UTF-8
chcp 65001 >nul
setlocal

:: 1. 定义 NDK 基础路径和编译器路径
set NDK_PATH=C:\Users\GT15\AppData\Local\Android\Sdk\ndk\30.0.14904198
set CXX=%NDK_PATH%\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android21-clang++.cmd

echo =======================================================
echo [1/3] 正在交叉编译算法引擎...
echo =======================================================
call %CXX% src\linux\main.cpp src\core\excavator_pipeline.cpp ^
    -I./src/core ^
    -I./3rdparty/rknn/include ^
    -I./3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/include ^
    -L./3rdparty/rknn/android/arm64-v8a -lrknnrt ^
    -L./3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/libs/arm64-v8a -lopencv_java4 ^
    -lc++_shared -lm -O3 -o test_excavator

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 编译失败！请检查上方报错信息。
    exit /b
)
echo ✅ 编译成功！

echo.
echo =======================================================
echo [2/3] 正在通过 ADB 同步文件到设备 (使用增量同步)...
echo =======================================================
:: 移除了所有不需要的 call 前缀，防止变量被底层错误展开
adb push --sync test_excavator /data/local/tmp/
adb shell chmod +x /data/local/tmp/test_excavator

adb push --sync %NDK_PATH%\toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so /data/local/tmp/
adb push --sync ./3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java4.so /data/local/tmp/
adb push --sync ./3rdparty/rknn/android/arm64-v8a/librknnrt.so /data/local/tmp/
adb push --sync ./tmp_files/best.rknn /data/local/tmp/
adb push --sync ./tmp_files/siamese_extractor.rknn /data/local/tmp/
adb push --sync ./tmp_files/test_video.mp4 /data/local/tmp/

echo.
echo =======================================================
echo [3/3] 正在安卓设备上运行测试...
echo =======================================================
:: 初始化并清空板端目录
adb shell "mkdir -p /data/local/tmp/out_frames && rm -f /data/local/tmp/out_frames/*.jpg"

:: 执行推理。没有 call 的干扰，%%04d 将被正确转义为 %04d 并传给 OpenCV
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && /data/local/tmp/test_excavator /data/local/tmp/best.rknn /data/local/tmp/siamese_extractor.rknn /data/local/tmp/test_video_fast.mp4 /data/local/tmp/out_frames/frame_%%04d.jpg"

echo.
echo =======================================================
echo 正在从开发板拉取渲染图片...
echo =======================================================
:: 规避了容易导致乱码中断的词汇，确保顺利打包拉回
adb pull /data/local/tmp/out_frames/ ./

echo.
echo ✅ 全部流程处理完毕！请在当前工程目录下的 out_frames 文件夹中查看结果。
echo =======================================================

endlocal
pause