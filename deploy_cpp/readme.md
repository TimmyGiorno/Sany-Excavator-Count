## 最终目录结构

```angular2html
deploy_cpp/
├── CMakeLists.txt                    # 统一构建 CMake 文件，通过 option 切换后端
├── include/
│   ├── types.h                       # DetectBox、ModelInfo 通用结构体
│   ├── inference_engine.h            # [新建] IBackend 抽象接口 + InferenceEngine 外观
│   ├── excavator_tracker.h           # 不变
│   └── yolo_engine.h                 # 保留（待后续清理）
├── src/
│   ├── inference_engine.cpp          # [新建] letterbox / decode / NMS（后端无关）
│   ├── excavator_tracker.cpp         # 不变
│   ├── main.cpp                      # 保留（Windows 测试入口）
│   └── backends/
│       ├── rknn_backend.cpp          # [新建] RKNN 特定代码，受 #ifdef USE_RKNN 保护
│       ├── mock_backend.cpp          # [新建] 空跑后端，分配假输出
│       └── onnx_backend.cpp          # [新建] ONNX Runtime 桩
├── jni/
│   ├── excavator_jni.cpp             # [更新] 引用 InferenceEngine + create_rknn_backend()
│   ├── rknn_api.h                    # 仅 Android 需要，留在这里
│   ├── ExcavatorNative.java          # 不变
│   ├── Android.mk                    # [更新] 指向新源文件路径 + -DUSE_RKNN
│   └── Application.mk                # 不变
├── scripts/
│   ├── build_windows.bat             # [新建] cmake -DUSE_MOCK=ON
│   ├── build_android.bat             # [移入] ndk-build
│   ├── build_android.sh              # [新建] Linux 版 ndk-build
│   ├── build_linux.sh                # [新建] cmake build_linux.sh rknn|mock|onnx
│   └── deploy_adb.bat                # [移入]
├── cmake/toolchains/                 # [新建] 预留自定义 toolchain
└── 3rdparty/                         # 第三方依赖库
```
