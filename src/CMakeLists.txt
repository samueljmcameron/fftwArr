file(GLOB LIB_SOURCES array3d.cpp array2d.cpp)


add_library(fftw_arr STATIC ${LIB_SOURCES})



target_include_directories(fftw_arr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fftw_arr PUBLIC MPI::MPI_CXX PkgConfig::FFTW-MPI ${FFTW_ARR_LIB})


set_target_properties(fftw_arr PROPERTIES PUBLIC_HEADER "array3d.hpp" PUBLIC_HEADER "array2d.hpp")


include(GNUInstallDirs)

install(TARGETS fftw_arr LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fftw_arr)


# generate pc file for pkg-config

set(target1 fftw_arr)
configure_file(fftw_arr.pc.in ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/fftw_arr.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/fftw_arr.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)


install(FILES ${CMAKE_SOURCE_DIR}/pkgconfig/fftw3mpi.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)


