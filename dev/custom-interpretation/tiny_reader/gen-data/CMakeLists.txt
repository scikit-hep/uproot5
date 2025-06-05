cmake_minimum_required(VERSION 3.20)

project(gen-data LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)

add_executable(gen-data
    src/main.cc
    src/TMyObject.cc
    src/TMySubObject.cc
)

target_include_directories(gen-data
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(gen-data
    PRIVATE
        ROOT::Core
        ROOT::RIO
        ROOT::Tree
)

ROOT_GENERATE_DICTIONARY(MyRootDict
        include/TMyObject.hh
        include/TMySubObject.hh
    LINKDEF include/LinkDef.h
    MODULE gen-data
)
