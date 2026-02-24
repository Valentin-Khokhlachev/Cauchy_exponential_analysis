cmake_minimum_required(VERSION 3.24)

project(Cauchy_exponential_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_COMPILE_OPTIONS "-Wall -Wextra -Wpedantic -O3 -march=native")

add_compile_definitions(SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(Eigen3_DIR "C:/msys64/mingw64/include/eigen3")
include_directories(${Eigen3_DIR})

set(Boost_DIR "C:/msys64/mingw64/include")
include_directories(${Boost_DIR})

set(GMP_INCLUDE_DIR "C:/msys64/mingw64/include")
set(GMP_LIB_DIR "C:/msys64/mingw64/lib")
set(MPFR_INCLUDE_DIR "C:/msys64/mingw64/include")
set(MPFR_LIB_DIR "C:/msys64/mingw64/lib")
set(MPC_INCLUDE_DIR "C:/msys64/mingw64/include")
set(MPC_LIB_DIR "C:/msys64/mingw64/lib")
set(OPENMP_INCLUDE_DIR "C:/msys64/mingw64/include")
set(OPENMP_LIB_DIR "C:/msys64/mingw64/lib")
set(OPENMP_LIB "gomp")



add_executable(${PROJECT_NAME}
    "main.cpp"
    "mpTypes.hpp"
    "exponential_analysis.hpp"
    "exponential_analysis.cpp"
    "settings.hpp"
)



target_include_directories(${PROJECT_NAME} PRIVATE ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} ${MPC_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PRIVATE ${GMP_LIB_DIR} ${MPFR_LIB_DIR} ${MPC_LIB_DIR})

target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PRIVATE ${OPENMP_LIB_DIR})



target_link_libraries(${PROJECT_NAME} PRIVATE mpc mpfr gmp gmpxx)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENMP_LIB})
target_include_directories(${PROJECT_NAME} PRIVATE ${Eigen3_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenMP_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)


