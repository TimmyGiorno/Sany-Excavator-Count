@echo off
REM ============================================================
REM  build_windows.bat -- Mock backend for pipeline benchmarking
REM  Usage: scripts\build_windows.bat
REM ============================================================
setlocal

set "PROJECT_DIR=%~dp0.."
set "BUILD_DIR=%PROJECT_DIR%\build_windows"

echo [INFO] Configuring CMake (Mock backend)...
cmake -B "%BUILD_DIR%" -S "%PROJECT_DIR%" -DUSE_MOCK=ON -DUSE_RKNN=OFF -DUSE_ONNX=OFF
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configure failed.
    exit /b 1
)

echo [INFO] Building...
cmake --build "%BUILD_DIR%" --config Release
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo   Build complete: %BUILD_DIR%\Release\test_video.exe
echo ============================================================
