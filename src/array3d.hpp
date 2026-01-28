#ifndef FFTWMPI_ARRAY3D_HPP
#define FFTWMPI_ARRAY3D_HPP

#include <fftw3-mpi.h>
#include <array>
#include <string>
#include <iostream>
#include <complex>
#include <memory>
#include <fstream>

#include "fftw_arr.hpp"

namespace fftwArr {

  
template < enum Transform rOc,typename T>
class array3D
{


private:
  T *arr;
  ptrdiff_t alloc_local, local_0_start;

  ptrdiff_t size;
  int nprocs,me;

  ptrdiff_t global_x_size;        // global size of the Nx (since it is not saved anywhere else)
  std::array<ptrdiff_t,3> sizeax; // local axis sizes of the array {nx,ny,nz} 
  
  std::string array_name;
  int spacer;
  MPI_Comm world;

  std::unique_ptr<fftwArr::array3D<rOc,T>> fftw_recv; // for I/O
  std::string operation_err_msg(const std::string &,
				const std::string &);

  
public:

  array3D();
  array3D(const MPI_Comm &,std::string,
	  ptrdiff_t, ptrdiff_t, ptrdiff_t);
  array3D(const array3D<rOc,T> &,std::string name = "");


  ~array3D();


  
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

  ptrdiff_t Nz() const
  {
    return sizeax[2];
  }


  ptrdiff_t xysize() const
  /* memory size of xy combined, accounting for array being non-contiguous
     (i.e. NOT just Nx*Ny) */
  {
    return sizeax[1]*spacer;
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
  

  T& operator()(ptrdiff_t , ptrdiff_t , ptrdiff_t );
  T operator()(ptrdiff_t , ptrdiff_t , ptrdiff_t ) const;

  T& operator()(ptrdiff_t );
  T operator()(ptrdiff_t ) const;


  array3D<rOc,T>& operator=(T other);
  array3D<rOc,T>& operator=(array3D<rOc,T> other);

  void reverseFlat(int,  int &, int &, int &) const;


  array3D<rOc,T>& operator*=(T rhs);
  array3D<rOc,T>& operator/=(T rhs);
  array3D<rOc,T>& operator+=(T rhs);
  array3D<rOc,T>& operator-=(T rhs);
  
  array3D<rOc,T>& operator*=(const array3D<rOc,T>& rhs);
  array3D<rOc,T>& operator/=(const array3D<rOc,T>& rhs);
  array3D<rOc,T>& operator+=(const array3D<rOc,T>& rhs);
  array3D<rOc,T>& operator-=(const array3D<rOc,T>& rhs);

  void write_to_binary(std::fstream &,
		       const bool overlap=true);


  void read_from_binary(std::fstream &,
			const bool overlap=true);

  
  /*
    void abs(array3D<rOc,T><double>&) const;
    void mod(array3D<rOc,T><double>&) const;
    void running_mod(array3D<rOc,T><double>&) const;
  */



  friend void swap(array3D<rOc,T>& first, array3D<rOc,T>& second)
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


template < enum fftwArr::Transform rOc, typename T>
std::ostream& operator<<(std::ostream& stream,
			 const fftwArr::array3D<rOc,T>& rhs)
{

  int nprocs = rhs.get_nprocs();
  int me = rhs.get_me();
  MPI_Comm world = rhs.get_world();
  
  if (me == 0)
    stream << "Proc\tnx\tny\tnz\t" << rhs.get_name() << std::endl;


  for (int p = 0; p < nprocs; p++) {
    if (p == me) {
      bool firstval = true;
      for (int nz = 0; nz < rhs.Nz(); nz++)
	for (int ny = 0; ny < rhs.Ny(); ny++)
	  for (int nx = 0; nx < rhs.Nx(); nx++)
	    if (firstval) {
	      stream << p << "\t" << nx << "\t"
		     << ny << "\t" << nz << "\t" << rhs(nx,ny,nz);
	      firstval = false;
	    } else
	      stream << std::endl << p << "\t" << nx << "\t"
		     << ny << "\t" << nz << "\t" << rhs(nx,ny,nz);
      
    }
    MPI_Barrier(world);
    
  }
  
  return stream;
}

#endif
