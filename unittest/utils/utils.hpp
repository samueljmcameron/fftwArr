#ifndef FFTWARR_TESTING_UTILS_HPP
#define FFTWARR_TESTING_UTILS_HPP


#include "fftw_arr/fftw_arr.hpp"


namespace fftwArrTestingUtils {


  template < typename T>
  int all_zero(T & array,double tolerance)
  {
    int local_flag = 0;
    
    for (int kz = 0; kz < array.Nz(); kz++)
      for (int jy = 0; jy < array.Ny(); jy++)
	for (int ix = 0; ix < array.Nx(); ix++)
	  if (std::abs(array(ix,jy,kz)) > tolerance)
	    local_flag += 1;

    return local_flag;
  }
  

  
  std::string SuccessMessage(const std::string &,enum fftwArr::Transform ,
			     int);
  std::string fftwArrName(const std::string &dtype,
			  enum fftwArr::Transform rOc,int dim);
  std::string TypeToString(const std::string &);
  std::string TransformToString(enum fftwArr::Transform );
}

#endif
