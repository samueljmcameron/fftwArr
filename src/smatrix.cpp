#include <complex>
#include <stdexcept>

#include <cstdint>

#include "smatrix.hpp"

using namespace fftwArr;

template <typename T>
sMatrix<T>& sMatrix<T>::operator= (const sMatrix & rhs)
{
  if (&rhs != this) {

    uncreate();

    create(rhs.begin(), rhs.end());

    _sizeax.at(0) = rhs.axis_size(0);
    _sizeax.at(1) = rhs.axis_size(1);
  }

  return *this;

}

template <typename T>
void sMatrix<T>::create()
{

  arr = limit = nullptr;
}


template <typename T>
void sMatrix<T>::create(size_t n, const T& val)
{
  arr = alloc.allocate(n);
  limit = arr + n;
  std::uninitialized_fill(arr, limit,val);

}

template <typename T>
void sMatrix<T>::create(const_iterator i, const_iterator j)
{

  arr = alloc.allocate(j - i);
  limit = std::uninitialized_copy(i,j,arr);
  
}

template <typename T>
void sMatrix<T>::uncreate()
{
  if (arr) {
    
    iterator it = limit;
    while (it != arr) {
      std::allocator_traits<std::allocator<T>>::destroy(alloc,--it);
    }
    alloc.deallocate(arr,limit-arr);

  }

  arr = limit = nullptr;
}


template <typename T>
void sMatrix<T>::resize(size_t Nx, size_t Ny)
{

  size_t new_size = Ny*Nx;

  iterator new_arr = alloc.allocate(new_size);
  
  uncreate();
  
  arr = new_arr;
  limit = new_arr+new_size;

  
  _sizeax.at(0) = Nx;
  _sizeax.at(1) = Ny;

  
}

template class fftwArr::sMatrix<double>;
template class fftwArr::sMatrix<std::complex<double>>;
