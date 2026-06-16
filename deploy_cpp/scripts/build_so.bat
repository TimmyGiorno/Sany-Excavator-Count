@echo off
:: 强制切换控制台编码为 UTF-8
chcp 65001 >nul
setlocal

set CMAKE_EXE="D:\CLion 2024.3.3\bin\cmake\win\x64\bin\cmake.exe"
set NINJA_EXE="D:\CLion 2024.3.3\bin\ninja\win\x64\ninja.exe"

:: 定义 NDK 路径
set NDK_PATH=C:\Users\GT15\AppData\Local\Android\Sdk\ndk\30.0.14904198

:: 回到工程根目录 (假设脚本在 scripts 目录下，这句会自动退回根目录)
cd /d "%~dp0\.."

:: 创建专门用于存放安卓编译产物的文件夹
if not exist build_android mkdir build_android
cd build_android

echo =======================================================
echo [1/2] 正在配置 Android NDK 交叉编译环境...
echo =======================================================
:: 使用 NDK 的 toolchain 强制以 Android 目标进行配置
%CMAKE_EXE% .. ^
    -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM=%NINJA_EXE% ^
    -DCMAKE_TOOLCHAIN_FILE="%NDK_PATH%\build\cmake\android.toolchain.cmake" ^
    -DANDROID_ABI=arm64-v8a ^
    -DANDROID_PLATFORM=android-21 ^
    -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ CMake 配置失败！请检查上方红字报错。
    pause
    exit /b
)

echo.
echo =======================================================
echo [2/2] 正在编译生成 libexcavator_jni.so ...
echo =======================================================
:: 执行多线程编译
cmake --build . --config Release -j 4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 编译链接失败！
    pause
    exit /b
)

echo.
echo =======================================================
echo 🎉 打包成功！
echo 请在 build_android 目录下寻找 libexcavator_jni.so 文件。
echo =======================================================
endlocal
pause