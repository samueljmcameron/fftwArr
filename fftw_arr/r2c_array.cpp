#include <typeinfo>
#include "r2c_array.hpp"

using namespace fftwArr;


template <typename T>
r2cArray<T>::r2cArray() : dimension(0)
/*
  Default constructor (does nothing).
*/
{
  arr=nullptr;
};


template <typename T>
r2cArray<T>::r2cArray(const r2cArray<T> & base,std::string name)
  : alloc_local(base.alloc_local),local_0_start(base.local_0_start),
    array_name(base.array_name), spacer(base.spacer),
    dimension(base.dimension),sizeax(base.sizeax)
/*
  Copy array, but if name (other than "") is provided then only make an
  array of the same size with the new name, but don't copy the elements in the
  array.

  Parameters
  ----------
  base : r2cArray
      The array to either copy from or set size structures from
  name : std::string (optional)
      The name of the new array, default is "" which makes this a true copy constructor.
      
*/


{

  ptrdiff_t tmpsize;
  
  if (typeid(T) == typeid(double)) {

    arr = (T*) fftw_alloc_real(2*alloc_local);

    tmpsize = 2*alloc_local;
    
  } else if (typeid(T) == typeid(std::complex<double>)) {
    arr = (T*) fftw_alloc_complex(alloc_local);
    tmpsize = alloc_local;
  } else
    throw std::runtime_error("array_r2c can only have type double or  std::complex<double>.");

  if (name != "") array_name = name;
  else std::copy(base.arr,base.arr+tmpsize,arr);
    

  
}


template <typename T>
r2cArray<T>::~r2cArray() {
  fftw_free(arr);
};

template <typename T>
void r2cArray<T>::assign(const MPI_Comm &comm,std::string name,
			 const std::vector<ptrdiff_t> lengths)
/*
  Constructor for a 3D array with axis sizes (iNz,iNy,iNx) (the x dimension
  varies the quickest). The array is not contiguous in memory for different
  y and z indices. Values are either all doubles, or all std::complex.

  Parameters
  ----------
  comm : mpi communicator
      Typically MPI_COMM_WORLD, but could be other I suppose.
  name : string
      The name of the array (useful when needing to save data).
      
      
*/
{
  
  dimension = lengths.size();
  if (dimension == 1)
    throw std::runtime_error("array_r2c must have length greater than 1.");

  sizeax = lengths;

  // temporarily (if not a complex typename) set sizeax to Nx/2 + 1
  sizeax[dimension-1] = lengths[dimension-1]/2 + 1;

  alloc_local = fftw_mpi_local_size_many(dimension, sizeax.data(),
					 1,FFTW_MPI_DEFAULT_BLOCK,
					 comm,&sizeax[0],&local_0_start);
  
  if (typeid(T) == typeid(double)) {
    sizeax[dimension-1] = lengths[dimension-1]; // reset sizeax[-1] to be Nx since real
    arr = (T*) fftw_alloc_real(2*alloc_local);  // see mpi 2d example in fftw doc
    spacer = 2*(lengths[dimension-1]/2+1);      // see mpi 2d example in fftw doc
    
  } else if (typeid(T) == typeid(std::complex<double>)) {
    spacer=sizeax[dimension-1];   
    arr = (T*) fftw_alloc_complex(alloc_local);
  } else
    throw std::runtime_error("r2cArray can only have type double, "
			     "or std::complex<double>.");
  
  array_name = name;
  setZero();  
  
};





template <typename T>
T& r2cArray<T>::operator()(const std::vector<ptrdiff_t> &indices)
/* read/write access array elements via indices vector. */
{

  ptrdiff_t flat = indices[0]*sizeax[dimension-2] + indices[1];

  for (int d = 1; d < dimension-2; d++)
    flat = flat*sizeax[dimension - 2 - d] + indices[d+1];

  flat *= spacer;
  flat += indices[dimension-1];
  
  return arr[flat];
}


template <typename T>
T r2cArray<T>::operator()(const std::vector<ptrdiff_t> & indices) const
/* read-only access (but don't change) array elements via indices vector. */
{
  ptrdiff_t flat = indices[0]*sizeax[dimension-2] + indices[1];

  for (int d = 1; d < dimension-2; d++)
    flat = flat*sizeax[dimension - 2 - d] + indices[d+1];

  flat *= spacer;
  flat += indices[dimension-1];
  
  return arr[flat];
}


template <typename T>
std::vector<ptrdiff_t> r2cArray<T>::unflatten_index(ptrdiff_t fake_flat) const
{

  std::vector<ptrdiff_t> indices(dimension);
  ptrdiff_t prefac = fake_flat;
  
  for (int d = dimension-1; d > 0; d--) {
    indices[d] = prefac % sizeax[d];
    prefac = prefac / sizeax[d];
  }

  indices[0] = prefac;

  return indices;

}

template <typename T>
ptrdiff_t r2cArray<T>::flatten_index(const std::vector<ptrdiff_t> &indices) const
{


  ptrdiff_t flat = indices[0]*sizeax[dimension-2] + indices[1];

  for (int d = 1; d < dimension-2; d++)
    flat = flat*sizeax[dimension - 2 - d] + indices[d+1];

  flat *= spacer;
  flat += indices[dimension-1];
  

  return flat;

}



template <typename T>
T& r2cArray<T>::operator()(ptrdiff_t fake_flat, std::vector<ptrdiff_t> & indices)
/* read/write access array elements via flattened array. */
{
  indices = unflatten_index(fake_flat);
  ptrdiff_t flat = flatten_index(indices);

  return arr[flat];
}


template <typename T>
T r2cArray<T>::operator()(ptrdiff_t fake_flat,std::vector<ptrdiff_t> & indices) const
/* read-only access (but don't change) array elements via flattened array. */
{
  indices = unflatten_index(fake_flat);
  ptrdiff_t flat = flatten_index(indices);

  return arr[flat];

}

template <typename T>
r2cArray<T>& r2cArray<T>::operator=(r2cArray<T> other)
{
  swap(*this,other);
  
  return *this;
}
//template class fftwArr::r2cArray<double>;
//template class fftwArr::r2cArray<std::complex<double>>;
