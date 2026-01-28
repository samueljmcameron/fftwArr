#ifndef FFTW_ARR_ARRAYS_HPP
#define FFTW_ARR_ARRAYS_HPP


#include <complex>
#include "array2d.hpp"
#include "array3d.hpp"


namespace  fftwArr {

  using r2c_2D = fftwArr::array2D<fftwArr::Transform::R2C,double>;
  using c2r_2D = fftwArr::array2D<fftwArr::Transform::C2R,std::complex<double>>;
  using c2c_2D = fftwArr::array2D<fftwArr::Transform::C2C,std::complex<double>>;


  
  
  using r2c_3D = fftwArr::array3D<fftwArr::Transform::R2C,double>;
  using c2r_3D = fftwArr::array3D<fftwArr::Transform::C2R,std::complex<double>>;
  using c2c_3D = fftwArr::array3D<fftwArr::Transform::C2C,std::complex<double>>;
  /*
  using fftwArr::r2c_2D = fftwArr::array2D<fftwArr::Transform::R2C,double>;
  using fftwArr::c2r_2D = fftwArr::array2D<fftwArr::Transform::C2R,std::complex<double>>;
  using fftwArr::c2c_2D = fftwArr::array2D<fftwArr::Transform::C2C,std::complex<double>>;


  
  
  using fftwArr::r2c_3D = fftwArr::array3D<fftwArr::Transform::R2C,double>;
  using fftwArr::c2r_3D = fftwArr::array3D<fftwArr::Transform::C2R,std::complex<double>>;
  using fftwArr::c2c_3D = fftwArr::array3D<fftwArr::Transform::C2C,std::complex<double>>;
  */
}


#endif
