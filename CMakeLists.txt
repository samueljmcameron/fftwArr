cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(fftwArr VERSION 1.1 DESCRIPTION "array interface for fftw3"
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

if (FFTW_MPI_BASE_DIR)
  configure_File(fftw3mpi.pc.in ${CMAKE_SOURCE_DIR}/pkgconfig/fftw3mpi.pc @ONLY)
else()
  message(FATAL_ERROR "must set FFTW_MPI_BASE_DIR to be equal to the location of the fftw3-mpi installation")
endif()

set(ENV{PKG_CONFIG_PATH} "${CMAKE_SOURCE_DIR}/pkgconfig")

pkg_check_modules(FFTW-MPI IMPORTED_TARGET REQUIRED fftw3mpi)



add_subdirectory(fftw_arr)
