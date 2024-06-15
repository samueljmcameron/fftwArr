#ifndef FFTWMPI_R2C2D_HPP
#define FFTWMPI_R2C2D_HPP

#include <fftw3-mpi.h>
#include <array>
#include <string>
#include <complex>
#include "r2c_array.hpp"

namespace fftwArr {

template <typename T>
class r2c2d : public r2cArray
{
private:
  
public:

  r2c2d(); // default constructor which does nothing

  r2c2d(const r2c2d<T> &,std::string name = ""); // copy constructor but with option to rename

  void assign(const MPI_Comm &,std::string, ptrdiff_t,ptrdiff_t); // actually allocates array


  T& operator()(ptrdiff_t, ptrdiff_t); // access element with indices
  T operator()(ptrdiff_t, ptrdiff_t ) const; // access element with indices

  
  T& operator()(ptrdiff_t ); // access element from flattened index
  T operator()(ptrdiff_t ) const; // access element from flattened index


  // element-wise operations
  virtual r2c2d<T>& operator/=(T rhs) override; 
  virtual void setZero() override;
  
  void abs(r2c2d<double>&) const; 
  void mod(r2c2d<double>&) const; 
  void running_mod(r2c2d<double>&) const;

  
  virtual ptrdiff_t xysize() const override;


};




};

#endif
