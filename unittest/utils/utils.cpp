#include <string>
#include <exception>
#include <complex>

#include "utils.hpp"

namespace fftwArrTestingUtils {

std::string SuccessMessage(const std::string &dtype,
			   enum fftwArr::Transform rOc,int dim)
{
 std::string message = "\nTesting for " + std::to_string(dim)
   + " fftwArr of type " + dtype  + " and " + TransformToString(rOc)
   + " was succesful.\n" ;

 return message;
}


std::string fftwArrName(const std::string &dtype,
			enum fftwArr::Transform rOc,int dim)
{

  return "fftwArr" + std::to_string(dim)  + "D_" + TransformToString(rOc)
    + "_" + dtype;

}

  

std::string TypeToString(const std::string &input)
{
  std::string output;
  if (input == typeid(double).name())
    output = "DOUBLE";
  else if (input == typeid(std::complex<double>).name())
    output = "COMPLEX_DOUBLE";
  else
    throw std::runtime_error("Type does not have an assigned name.");

  return output;
  
}

std::string TransformToString(enum fftwArr::Transform rOc)
{

  std::string name;
  if (rOc ==  fftwArr::Transform::C2C)
    name = "c2c";
  else if (rOc ==  fftwArr::Transform::R2C)
    name = "r2c";
  else if (rOc ==  fftwArr::Transform::C2R)
    name = "c2r";
  else
    throw std::runtime_error("Transform does not have an assigned name.");
  return name;
}

}
