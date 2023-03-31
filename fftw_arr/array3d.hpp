#ifndef FFTWMPI_ARRAY3D_HPP
#define FFTWMPI_ARRAY3D_HPP

#include <fftw3-mpi.h>
#include <array>
#include <string>
#include <complex>

namespace fftwArr {

template <typename T>
class array3D
{
private:
  T *arr;
  ptrdiff_t alloc_local, local_0_start;

  ptrdiff_t size;

  std::array<ptrdiff_t,3> sizeax; // = {Nz,Ny,Nx} because for some stupid reason 12 months
  //                                  ago I thought x index should vary fastest
  
  
  std::string array_name;
  int spacer;

  
public:

  array3D(const MPI_Comm &,std::string,
	  ptrdiff_t, ptrdiff_t, ptrdiff_t);
  array3D(const fftw_MPI_3Darray<T> &,std::string name = "");


  void reverseFlat(int,  int &, int &, int &) const;


  
  ~array3D();



  
  T* data() {
    return arr;
  };

  ptrdiff_t totalsize() const {
    return size;
  };


  ptrdiff_t Nz() const
  {
    return sizeax[0];
  }

  ptrdiff_t Ny() const
  {
    return sizeax[1];
  }

  ptrdiff_t Nx() const
  {
    return sizeax[2];
  }


  
  

  ptrdiff_t get_local0start() const {
    return local_0_start;
  };

  std::string get_name() const {
    return array_name;
  }
    
  T& operator()(ptrdiff_t , ptrdiff_t , ptrdiff_t );
  T operator()(ptrdiff_t , ptrdiff_t , ptrdiff_t ) const;

  T& operator()(ptrdiff_t );
  T operator()(ptrdiff_t ) const;

  array3D<T>& operator=(array3D<T> other);
  array3D<T>& operator/=(T rhs);


  
  void abs(array3D<double>&) const;
  void mod(array3D<double>&) const;

  void setZero();

  void running_mod(array3D<double>&) const;

  friend void swap(array3D<T>& first, array3D<T>& second)
  {
    using std::swap;

    swap(first.arr,second.arr);
    swap(first.alloc_local,second.alloc_local);
    swap(first.local_0_start,second.local_0_start);
    swap(first.size,second.size);
    swap(first.sizeax,second.sizeax);
    swap(first.array_name,second.array_name);
    swap(first.spacer,second.spacer);

    return;

  }

  


};




};

#endif
