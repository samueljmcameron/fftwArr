#ifndef FFTWARR_TESTING_UTILS_HPP
#define FFTWARR_TESTING_UTILS_HPP


#include "fftw_arr/fftw_arr.hpp"


namespace fftwArrTestingUtils {

  std::string SuccessMessage(const std::string &,enum fftwArr::Transform ,
			     int);
  std::string fftwArrName(const std::string &dtype,
			  enum fftwArr::Transform rOc,int dim);
  std::string TypeToString(const std::string &);
  std::string TransformToString(enum fftwArr::Transform );
}

#endif
