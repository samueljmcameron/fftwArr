#ifndef FFTWARR_TESTING_UTILS_HPP
#define FFTWARR_TESTING_UTILS_HPP


#include "fftw_arr/fftw_arr.hpp"


namespace fftwArrTestingUtils {


  template < typename T>
  int all_zero_3d(T & array,double tolerance)
  {
    int local_flag = 0;
    
    for (int kz = 0; kz < array.size_axis2(); kz++)
      for (int jy = 0; jy < array.size_axis1(); jy++)
	for (int ix = 0; ix < array.size_axis0(); ix++)
	  if (std::abs(array(ix,jy,kz)) > tolerance)
	    local_flag += 1;

    return local_flag;
  }
  

  template < typename T>
  int all_zero_2d(T & array,double tolerance)
  {
    int local_flag = 0;
    
    for (int jy = 0; jy < array.size_axis1(); jy++)
      for (int ix = 0; ix < array.size_axis0(); ix++)
	if (std::abs(array(ix,jy)) > tolerance)
	  local_flag += 1;

    return local_flag;
  }

  
  
  std::string SuccessMessage(const std::string &,enum fftwArr::Transform ,
			     int,bool);
  std::string fftwArrName(const std::string &,
			  enum fftwArr::Transform ,int ,bool);
  std::string TypeToString(const std::string &);
  std::string TransformToString(enum fftwArr::Transform );
}

#endif
