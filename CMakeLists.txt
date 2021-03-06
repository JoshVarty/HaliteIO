cmake_minimum_required(VERSION 2.8)

project(MyBot)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wno-unused-function -pedantic")

include_directories(${CMAKE_SOURCE_DIR}/hlt)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

foreach(dir ${dirs})
    file(GLOB_RECURSE SOURCE ${dir}/*.[ch]*)
    set(SOURCE_FILES "${SOURCE_FILES}" ${SOURCE})
endforeach()

include_directories(${CMAKE_SOURCE_DIR})
set(SOURCE_FILES "${SOURCE_FILES}" MyBot.cpp)

add_executable(MyBot ${SOURCE_FILES})
target_link_libraries(MyBot "${TORCH_LIBRARIES}")

if(MINGW)
    target_link_libraries(MyBot -static)
endif()
