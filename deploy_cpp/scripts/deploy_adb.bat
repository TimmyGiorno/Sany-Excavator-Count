@echo off
REM ============================================================
REM  deploy_adb.bat
REM  Push excavator_algo.so + model to RK3568 Android device via ADB
REM
REM  Prerequisites:
REM    1. ADB installed and in PATH
REM    2. Device connected via USB and ADB debugging enabled
REM    3. Device is rooted or target dir is writable
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0..~dp0"
set "SO_DIR=%PROJECT_DIR%output\arm64-v8a"
set "MODEL_DIR=%PROJECT_DIR%..\_files\models"

REM ---- Device paths ----
set "DEVICE_LIB=/data/local/tmp"
set "DEVICE_MODEL=/sdcard/excavator"

REM ---- Check ADB ----
adb devices 2>nul | findstr /r "^[0-9a-f]" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] No ADB device found. Connect your RK3568 and enable USB debugging.
    exit /b 1
)
echo [INFO] ADB device detected.

REM ---- Push .so files ----
echo.
echo [INFO] Pushing .so libraries to %DEVICE_LIB%...
for %%f in ("%SO_DIR%\*.so") do (
    echo   Pushing %%~nxf...
    adb push "%%f" "%DEVICE_LIB%/%%~nxf"
)

REM ---- Push model ----
echo.
if exist "%MODEL_DIR%\best.rknn" (
    echo [INFO] Pushing RKNN model...
    adb shell mkdir -p %DEVICE_MODEL%
    adb push "%MODEL_DIR%\best.rknn" "%DEVICE_MODEL%/best.rknn"
) else (
    echo [WARN] best.rknn not found in _files\models\
    echo   Convert it first: python rknn_sim_deploy\convert_video_script.py
)

REM ---- Set permissions ----
echo.
echo [INFO] Setting permissions...
adb shell chmod 755 "%DEVICE_LIB%/libexcavator_algo.so"
adb shell chmod 755 "%DEVICE_LIB%/libc++_shared.so"

echo.
echo ============================================================
echo   Deploy complete!
echo.
echo   Library: %DEVICE_LIB%/libexcavator_algo.so
echo   Model:   %DEVICE_MODEL%/best.rknn
echo.
echo   On the Android app side, load with:
echo     System.load("%DEVICE_LIB%/libexcavator_algo.so");
echo.
echo   Or if deploying to /system/lib64 (requires root):
echo     adb root
echo     adb remount
echo     adb push output\arm64-v8a\*.so /system/lib64/
echo ============================================================
