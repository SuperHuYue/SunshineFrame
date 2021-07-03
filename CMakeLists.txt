cmake_minimum_required(VERSION 3.6)
project(DeepTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include_directories(${CMAKE_SOURCE_DIR}/layers ${CMAKE_SOURCE_DIR})
file(GLOB LAYER ${CMAKE_SOURCE_DIR}/layers/*.hpp)
add_executable(${PROJECT_NAME} main.cpp CMatrix.hpp CMatrix.cpp SunshineNet.cpp SunshineNet.hpp ${LAYER})
target_link_libraries(${PROJECT_NAME} gtest_main)
#target_link_libraries(${PROJECT_NAME} stdc++fs)
include("GoogleTest")
gtest_discover_tests(${PROJECT_NAME})
message(STATUS "end...")



