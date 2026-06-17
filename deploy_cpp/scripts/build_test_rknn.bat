@echo off

:: 强制切换控制台编码为 UTF-8
chcp 65001 >nul
setlocal

:: 1. 定义 NDK 基础路径和编译器路径 (保持你的原始路径)
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
    -lc++_shared -lm -O3 -o ./tmp_files/test_excavator

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 编译失败！请检查上方报错信息。
    exit /b
)
echo ✅ 编译成功！

echo.
echo =======================================================
echo [2/3] 正在通过 ADB 同步文件到设备...
echo =======================================================
adb push --sync ./tmp_files/test_excavator /data/local/tmp/
adb shell chmod +x /data/local/tmp/test_excavator

adb push --sync %NDK_PATH%\toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\aarch64-linux-android\libc++_shared.so /data/local/tmp/
adb push --sync ./3rdparty/opencv/opencv-4.13.0-android-sdk/OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java4.so /data/local/tmp/
adb push --sync ./3rdparty/rknn/android/arm64-v8a/librknnrt.so /data/local/tmp/
adb push --sync ./tmp_files/best_320.rknn /data/local/tmp/
adb push --sync ./tmp_files/test_video_shift_fast.mp4 /data/local/tmp/

echo.
echo =======================================================
echo [3/3] 正在安卓设备上运行分流测试...
echo =======================================================
:: 初始化并清空板端渲染图输出目录
adb shell "mkdir -p /data/local/tmp/out_frames && rm -f /data/local/tmp/out_frames/*.jpg"

echo.
echo -------------------------------------------------------
echo ▶ 执行测试 1：常规视频流推理 (每100帧输出性能耗时)
echo -------------------------------------------------------
:: 常规测试传入 3 个参数：<yolo模型> <视频> <test_mode=1>
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && /data/local/tmp/test_excavator /data/local/tmp/best_320.rknn /data/local/tmp/test_video_shift_fast.mp4 1 /data/local/tmp/out_frames/frame_%%04d.jpg"

echo.
echo -------------------------------------------------------
echo ▶ 执行测试 2：纯断电恢复状态流测试
echo -------------------------------------------------------
:: 一键测试开机直接从断电中恢复的场景 (对应 test_mode=2)
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && /data/local/tmp/test_excavator /data/local/tmp/best_320.rknn /data/local/tmp/test_video_shift_fast.mp4 2"

echo.
echo -------------------------------------------------------
echo ▶ 执行测试 3：纯挂机/业务超时预警测试 (触发单次推送)
echo -------------------------------------------------------
:: 一键测试正常作业中突然停工卡死12秒的超时上报场景 (对应 test_mode=3)
adb shell "export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && /data/local/tmp/test_excavator /data/local/tmp/best_320.rknn /data/local/tmp/test_video_shift_fast.mp4 3"

echo.
echo =======================================================
echo 正在从开发板拉取渲染图片...
echo =======================================================
adb pull /data/local/tmp/out_frames/ ./tmp_files/

echo.
echo ✅ 全部流程处理完毕！
echo    1. 检查控制台日志：模式 1 看性能FPS、模式 2 看断电恢复、模式 3 看超时预警 (JSON格式)。
echo    2. 检查电脑端 ./tmp_files/out_frames 查看框选渲染图。
echo =======================================================

endlocal
pause