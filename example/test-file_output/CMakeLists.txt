cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(fftwArrExample VERSION 1.0 DESCRIPTION "Testing file_output capabilities."
        LANGUAGES CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()


find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

find_package(PkgConfig REQUIRED)

set(PKG_CHECK_MODULES_STATIC)

set(CMAKE_FIND_LIBRARY_SUFFIXES .a)


MESSAGE(STATUS "CMAKE_PREFIX_PATH:" ${CMAKE_PREFIX_PATH})
MESSAGE(STATUS "CMAKE_FRAMEWORK_PATH:" ${CMAKE_FRAMEWORK_PATH})
MESSAGE(STATUS "CMAKE_APPBUNDLE_PATH:" ${CMAKE_APPBUNDLE_PATH})

pkg_check_modules(FFTW-MPI IMPORTED_TARGET fftw3mpi)
if (NOT FFTW-MPI_FOUND)
  message(FATAL_ERROR "Need to specify where to find fftw3_mpi in CMAKE_PREFIX_PATH")
endif()
MESSAGE(STATUS "FFTW-MPI_LIBRARIES::" ${FFTW-MPI_LIBRARIES})




MESSAGE(STATUS "CMAKE_PREFIX_PATH:" ${CMAKE_PREFIX_PATH})
MESSAGE(STATUS "CMAKE_FRAMEWORK_PATH:" ${CMAKE_FRAMEWORK_PATH})
MESSAGE(STATUS "CMAKE_APPBUNDLE_PATH:" ${CMAKE_APPBUNDLE_PATH})
pkg_check_modules(FFTWARR IMPORTED_TARGET REQUIRED fftw_arr)

MESSAGE(STATUS "FFTWARR_LIBRARIES::" ${FFTWARR_LIBRARIES})


add_executable(test-file_output main.cpp)

target_link_libraries(test-file_output PRIVATE MPI::MPI_CXX PkgConfig::FFTW-MPI PkgConfig::FFTWARR)


MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
