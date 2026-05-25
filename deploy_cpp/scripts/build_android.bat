@echo off
REM ============================================================
REM  build_android.bat -- ndk-build for RK3568 Android
REM  Usage: scripts\build_android.bat
REM ============================================================
setlocal enabledelayedexpansion

if "%NDK_HOME%"=="" (
    set NDK_HOME=C:\Users\GT15\AppData\Local\Android\Sdk\ndk\30.0.14904198
)

set "PROJECT_DIR=%~dp0.."
set "JNI_DIR=%PROJECT_DIR%\jni"
set "OUTPUT_DIR=%PROJECT_DIR%\output\arm64-v8a"

if not exist "%PROJECT_DIR%\3rdparty\rknn\librknnrt.so" (
    echo [WARN] librknnrt.so not found in 3rdparty\rknn\. Pull from device:
    echo   adb pull /vendor/lib64/librknnrt.so deploy_cpp\3rdparty\rknn\
    echo.
)

echo [INFO] Building with ndk-build...
pushd "%JNI_DIR%"
call "%NDK_HOME%\ndk-build.cmd" NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk
set BR=%ERRORLEVEL%
popd
if %BR% neq 0 exit /b %BR%

mkdir "%OUTPUT_DIR%" 2>nul

copy /Y "%PROJECT_DIR%\libs\arm64-v8a\libexcavator_algo.so" "%OUTPUT_DIR%\"

copy /Y "%PROJECT_DIR%\obj\local\arm64-v8a\test_excavator" "%OUTPUT_DIR%\"

copy /Y "%PROJECT_DIR%\libs\arm64-v8a\libc++_shared.so" "%OUTPUT_DIR%\"

echo.
echo ============================================================
echo   Build complete: %OUTPUT_DIR%
echo   Files:
echo     libexcavator_algo.so  (JNI library)
echo     test_excavator        (adb shell test tool)
echo     libc++_shared.so
echo ============================================================