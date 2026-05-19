cmake_minimum_required(VERSION 3.10)
project(SanyZLExcavatorTracker)

option(USE_RKNN_NPU "Use RKNN Hardware Acceleration" ON)

find_library(log-lib log)