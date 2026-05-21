# RK3568 is ARMv8-A (arm64-v8a)
APP_ABI := arm64-v8a

# Android 10+ (API 29). RK3568 boards typically run Android 11/12 (API 30/31)
APP_PLATFORM := android-29

# Use shared C++ runtime (libc++_shared.so must be bundled)
APP_STL := c++_shared

# Build release by default
APP_OPTIM := release

# Enable exceptions + RTTI
APP_CPPFLAGS += -frtti -fexceptions
