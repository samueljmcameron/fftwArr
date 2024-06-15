#ifndef FFTWMPI_R2CARRAY_HPP
#define FFTWMPI_R2CARRAY_HPP

#include <fftw3-mpi.h>
#include <array>
#include <string>
#include <complex>

namespace fftwArr {

template <typename T>
class r2cArray
{
private:
  T *arr;                        // pointer to array data
  ptrdiff_t alloc_local;         // size of the locally allocated array (including all junk memory)
  ptrdiff_t local_0_start;       // the global index that the local array starts at
  std::vector<ptrdiff_t> sizeax; // shape of array according to outside user (non-contiguous)
  
  int dimension;
  std::string array_name;
  int spacer;
  
public:

  r2cArray(); // default constructor which does nothing
  r2cArray(const MPI_Comm &,std::string,
	   const std::vector<ptrdiff_t> ); // second constructor which constructs array
  r2cArray(const r2cArray<T> &,std::string name = ""); // copy constructor but with option to rename
  ~r2cArray();

  void assign(const MPI_Comm &,std::string,
	      const std::vector<ptrdiff_t>); // actually allocates array


  std::vector<ptrdiff_t> unflatten_index(ptrdiff_t) const; // get indices unflattened
  ptrdiff_t flatten_index(const std::vector<ptrdiff_t> &indices) const; // get flattened index
  T& operator()(const std::vector<ptrdiff_t> & ); // access element from vector 
  T operator()(const std::vector<ptrdiff_t> & ) const; // access element from flattened index

  
  T& operator()(ptrdiff_t ); // access element from flattened index
  T operator()(ptrdiff_t ) const; // access element from flattened index

  r2cArray<T>& operator=(r2cArray<T> other); // assign operator


  // element-wise operations

  virtual r2cArray<T>& operator/=(T rhs) = 0; // division operator
  virtual void setZero() = 0; // set all array components to zero

  
  T* data() { return arr;};
  T* data() const { return arr;};
  virtual ptrdiff_t xysize() const = 0;

  
  ptrdiff_t get_local0start() const {return local_0_start;};
  std::string get_name() const {return array_name;};
  ptrdiff_t get_sizeax(int dim) { return sizeax.at(dim)};


  friend void swap(r2cArray<T>& first, r2cArray<T>& second)
  {
    using std::swap;

    swap(first.arr,second.arr);
    swap(first.alloc_local,second.alloc_local);
    swap(first.local_0_start,second.local_0_start);
    swap(first.sizeax,second.sizeax);
    swap(first.array_name,second.array_name);
    swap(first.spacer,second.spacer);
    swap(first.dimension,second.dimension);

    return;

  }

  


};




};

#endif
