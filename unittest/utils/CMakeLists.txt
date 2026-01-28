cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(fftwArrTestingUtils VERSION 1.0 DESCRIPTION "Utility function library."
        LANGUAGES CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()


find_package(PkgConfig REQUIRED)

set(PKG_CHECK_MODULES_STATIC)

set(CMAKE_FIND_LIBRARY_SUFFIXES .a)


pkg_check_modules(FFTWARR IMPORTED_TARGET REQUIRED fftw_arr)

MESSAGE(STATUS "FFTWARR_LIBRARIES::" ${FFTWARR_LIBRARIES})


add_library(fftw_arr_testing_utils utils.cpp)

target_include_directories(fftw_arr_testing_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fftw_arr_testing_utils PUBLIC PkgConfig::FFTWARR)

set_target_properties(fftw_arr_testing_utils PROPERTIES PUBLIC_HEADER "utils.hpp")

include(GNUInstallDirs)

install(TARGETS fftw_arr_testing_utils LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fftw_arr_testing_utils)

set(target1 fftw_arr_testing_utils)
configure_file(fftw_arr_testing_utils.pc.in ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/fftw_arr_testing_utils.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/fftw_arr_testing_utils.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)



MESSAGE(STATUS "Compiler flags:" ${CMAKE_CXX_COMPILE_FLAGS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
