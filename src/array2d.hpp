#ifndef FFTWMPI_ARRAY2D_HPP
#define FFTWMPI_ARRAY2D_HPP

#include <fftw3-mpi.h>
#include <array>
#include <string>
#include <iostream>
#include <complex>
#include <memory>
#include <fstream>

namespace fftwArr {

template <typename T>
class array2D
{

  friend class Conjugate;
private:
  T *arr;
  ptrdiff_t alloc_local, local_0_start;

  ptrdiff_t size;
  int nprocs,me;

  ptrdiff_t global_x_size;        // global size of the Nx (since it is not saved anywhere else)
  std::array<ptrdiff_t,2> sizeax; // local axis sizes of the array {nx,ny,nz} 
  
  std::string array_name;
  int spacer;
  MPI_Comm world;

  std::unique_ptr<fftwArr::array2D<T>> fftw_recv; // for I/O
  std::string operation_err_msg(const std::string &,
				const std::string &);
  
public:

  array2D();
  array2D(const MPI_Comm &,std::string,
	  ptrdiff_t, ptrdiff_t);
  array2D(const array2D<T> &,std::string name = "");


  void assign(const MPI_Comm &,std::string,
	      ptrdiff_t, ptrdiff_t);


  ~array2D();

  
  T* data() {
    return arr;
  };
  T* data() const {
    return arr;
  };

  

  ptrdiff_t totalsize() const {
    return size;
  };


  ptrdiff_t Nx() const
  {
    return sizeax[0];
  }

  ptrdiff_t Ny() const
  {
    return sizeax[1];
  }


  ptrdiff_t xsize() const
  /* memory size of x, accounting for array being non-contiguous
     (i.e. NOT just Nx) */
  {
    return spacer;
  }
  

  ptrdiff_t get_local0start() const {
    return local_0_start;
  };

  std::string get_name() const {
    return array_name;
  }


  int get_nprocs() const {
    return nprocs;
  }
  
  int get_me() const {
    return me;
  }
  
  MPI_Comm get_world() const {
    return world;
  }
  

  T& operator()(ptrdiff_t , ptrdiff_t );
  T operator()(ptrdiff_t , ptrdiff_t ) const;

  T& operator()(ptrdiff_t );
  T operator()(ptrdiff_t ) const;


  array2D<T>& operator=(T other);
  array2D<T>& operator=(array2D<T> other);

  void reverseFlat(int,  int &, int &) const;


  array2D<T>& operator*=(T rhs);
  array2D<T>& operator/=(T rhs);
  array2D<T>& operator+=(T rhs);
  array2D<T>& operator-=(T rhs);
  
  array2D<T>& operator*=(const array2D<T>& rhs);
  array2D<T>& operator/=(const array2D<T>& rhs);
  array2D<T>& operator+=(const array2D<T>& rhs);
  array2D<T>& operator-=(const array2D<T>& rhs);

  void write_to_binary(std::fstream &,
		       const bool overlap=true);


  void read_from_binary(std::fstream &,
			const bool overlap=true);

  
  /*
    void abs(array2D<T><double>&) const;
    void mod(array2D<T><double>&) const;
    void running_mod(array2D<T><double>&) const;
  */



  friend void swap(array2D<T>& first, array2D<T>& second)
  {
    using std::swap;

    swap(first.arr,second.arr);
    swap(first.alloc_local,second.alloc_local);
    swap(first.local_0_start,second.local_0_start);
    swap(first.size,second.size);
    swap(first.sizeax,second.sizeax);
    swap(first.array_name,second.array_name);
    swap(first.spacer,second.spacer);
    swap(first.me,second.me);
    swap(first.nprocs,second.nprocs);
    swap(first.world,second.world);
    
    return;

  }
  
  
};




};


template <typename T>
std::ostream& operator<<(std::ostream& stream,
			 const fftwArr::array2D<T>& rhs)
{

  int nprocs = rhs.get_nprocs();
  int me = rhs.get_me();
  MPI_Comm world = rhs.get_world();
  
  if (me == 0)
    stream << "Proc\tnx\tny\t" << rhs.get_name() << std::endl;


  for (int p = 0; p < nprocs; p++) {
    if (p == me) {
      bool firstval = true;
      for (int ny = 0; ny < rhs.Ny(); ny++)
	for (int nx = 0; nx < rhs.Nx(); nx++)
	  if (firstval) {
	    stream << p << "\t" << nx << "\t"
		   << ny << "\t" << rhs(nx,ny);
	    firstval = false;
	  } else
	    stream << std::endl << p << "\t" << nx << "\t"
		   << ny << "\t" << rhs(nx,ny);
      
    }
    MPI_Barrier(world);
    
  }
  
  return stream;
}

#endif
