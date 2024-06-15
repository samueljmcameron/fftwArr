#include <typeinfo>
#include "r2c_2d.hpp"

using namespace fftwArr;


template <typename T>
r2c2d<T>::r2c2d() : r2cArray()
/*
  Default constructor (does nothing).
*/
{
};


template <typename T>
r2c2d<T>::r2c2d(const MPI_Comm &comm,std::string name,ptrdiff_t Nx,
		ptrdiff_t Ny) : r2cArray(comm,name,{Nx,Ny})
{
  
};


template <typename T>
T& r2c2d<T>::operator()(ptrdiff_t i, ptrdiff_t j)
/* read/write access array elements via indices vector. */
{

  return arr[j + i*spacer];
}


template <typename T>
T& r2c2d<T>::operator()(ptrdiff_t i, ptrdiff_t j) const
/* read-only access (but don't change) array elements via indices vector. */
{

  return arr[j + i*spacer];
}


template <typename T>
T& r2c2d<T>::operator()(ptrdiff_t flat)
/* read/write access array elements via flattened array. */
{

  int j = flat % sizeax[1];
  int i = (flat/sizeax[1]);
  

  return arr[j + i*spacer];
}


template <typename T>
T r2c2d<T>::operator()(ptrdiff_t flat) const
/* read-only access (but don't change) array elements via flattened array. */
{

  int j = flat % sizeax[1];
  int i = (flat/sizeax[1]);
  
  return arr[j + i*spacer];

}


template <typename T>
void r2c2d<T>::setZero()
{

  for (int i = 0; i < sizeax[0]; i++) {
    for (int j = 0; j < sizeax[1]; j++) {
      arr[j + i * spacer] = 0.0;
    }
  }

  return;

}


template <typename T>
void r2c2d<T>::abs(r2c2d<double>& modulus) const
{


    
  for (int i = 0; i < 2; i++)
    if (sizeax[i] != modulus.get_sizeax(i))
      throw std::runtime_error("Cannot take abs of r2c2d (wrong output shape).");


  for (int i = 0; i < sizeax[0]; i++) {
    for (int j = 0; j < sizeax[1]; j++) {
      modulus(i,j) = std::abs(arr[j + i * spacer]);
    }
  }

  return;

}


template <typename T>
void r2c2d<T>::mod(r2c2d<double>& modulus) const
{


    
  for (int i = 0; i < 2; i++)
    if (sizeax[i] != modulus.get_sizeax(i))
      throw std::runtime_error("Cannot take modulus of r2c2d (wrong output shape).");


  for (int i = 0; i < sizeax[0]; i++) {
    for (int j = 0; j < sizeax[1]; j++) {
      modulus(i,j) = std::abs(arr[j + i * spacer])*std::abs(arr[j + i * spacer]);
    }
  }

  return;

}

template <typename T>
void r2c2d<T>::running_mod(r2c2d<double>& modulus) const
{


    
  for (int i = 0; i < 2; i++)
    if (sizeax[i] != modulus.get_sizeax(i))
      throw std::runtime_error("Cannot take modulus of r2c2d (wrong output shape).");


  for (int i = 0; i < sizeax[0]; i++) {
    for (int j = 0; j < sizeax[1]; j++) {
      modulus(i,j) += std::abs(arr[j + i * spacer])*std::abs(arr[j + i * spacer]);
    }
  }

  return;

}






template class r2c2d<double>;
template class r2c2d<std::complex<double>>;

// Template below doesn't work because it forces return of double [2] array.
//template class r2c2d<fftw_complex>; 
