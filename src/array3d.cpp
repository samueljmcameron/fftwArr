#include "array3d.hpp"

using namespace fftwArr;

template <typename T>
array3D<T>::array3D()
/*
  Default constructor (does nothing).
*/
{
  arr=nullptr;
};


template <typename T>
array3D<T>::array3D(const MPI_Comm &comm,std::string name,
			 ptrdiff_t Nx, ptrdiff_t Ny, ptrdiff_t Nz)
/*
  Constructor for a 3D array with axis sizes (Nx,Ny,Nz) (the x dimension
  varies the quickest). The array is not contiguous in memory for different
  y and z indices.

  The structure (once parallelised will be):

  y
  |  x
  | /
  |/
   ------ z
  
        _______    _______    _______           _______
       /      /|  /      /|  /      /|         /      /|
  Nx  /      / | /      / | /      / |        /      / | 
     /      /  |/      /  |/      /  |       /      /  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |  ...  |      |  |
     |      |  ||      |  ||      |  |       |      |  |
 Ny  |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  /|      |  /|      |  /       |      |  /
     |      | / |      | / |      | /        |      | /
     |______|/  |______|/  |______|/         |______|/

       Nz_1       Nz_1       Nz_2       ...    Nz_p


  Parameters
  ----------
  comm : mpi communicator
      Typically MPI_COMM_WORLD, but could be other I suppose.
  name : string
      The name of the array (useful when needing to save data).
  Nx : ptrdiff_t
      The (global) number of points in the x direction
  Ny : ptrdiff_t
      The (global) number of points in the y direction
  Nz : ptrdiff_t
      The (global) number of points in the z direction
*/
{


  ptrdiff_t local_n0;
  
  world = comm;

  // pass axes along so that z varies most slowly, then y, then x
  alloc_local = fftw_mpi_local_size_3d(Nz, Ny , Nx/2 + 1,
				       world,&local_n0,&local_0_start);
  

  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  
  sizeax[2] = local_n0;
  sizeax[1] = Ny;
  global_x_size = Nx;

  if (typeid(T) == typeid(double)) {
    sizeax[0] = Nx;
    arr = (T*) fftw_alloc_real(2*alloc_local);
    spacer = 2*(Nx/2+1);    
    size = spacer*alloc_local;
  } else if (typeid(T) == typeid(std::complex<double>)) {
    sizeax[0] = Nx/2+1;
    arr = (T*) fftw_alloc_complex(alloc_local);
    spacer=Nx/2+1;
    size = alloc_local;
  } else
    throw std::runtime_error("array3D can only have type double, "
			     "or std::complex<double>.");
  
    
  array_name = name;

  // the next line sets all elements of the array equal to zero
  *this = 0;
};





template <typename T>
array3D<T>::array3D(const array3D<T> & base,std::string name)
  : alloc_local(base.alloc_local),local_0_start(base.local_0_start),
    size(base.size),array_name(base.array_name), spacer(base.spacer),
    global_x_size(base.global_x_size),nprocs(base.nprocs),me(base.me),world(base.world)
/*
  Copy array, but if name (other than "") is provided then only make an
  array of the same size with the new name, but don't copy the elements in the
  array.

  Parameters
  ----------
  base : array3D
      The array to either copy from or set size structures from
  name : std::string (optional)
      The name of the new array, default is "" which makes this a true copy constructor.
      
*/


{


  sizeax[0] = base.sizeax[0];
  sizeax[1] = base.sizeax[1];
  sizeax[2] = base.sizeax[2];


  

  ptrdiff_t tmpsize;
  
  if (typeid(T) == typeid(double)) {
    arr = (T*) fftw_alloc_real(2*alloc_local);

    tmpsize = 2*alloc_local;
  } else if (typeid(T) == typeid(std::complex<double>)) {
    arr = (T*) fftw_alloc_complex(alloc_local);
    tmpsize = alloc_local;

  } else
    throw std::runtime_error("array3D can only have type double, "
			     "or std::complex<double>.");

    
  if (name != "") array_name = name;
  else std::copy(base.arr, base.arr + tmpsize,arr);

  
}


template <typename T>
void array3D<T>::assign(const MPI_Comm &comm,std::string name,
			 ptrdiff_t Nx, ptrdiff_t Ny, ptrdiff_t Nz)
/*

  Parameters
  ----------
  comm : mpi communicator
      Typically MPI_COMM_WORLD, but could be other I suppose.
  name : string
      The name of the array (useful when needing to save data).
      
      
*/
{
  ptrdiff_t local_n0;
  
  world = comm;
  alloc_local = fftw_mpi_local_size_3d(Nz, Ny , Nx/2 + 1,
				       world,&local_n0,&local_0_start);
  
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);
  
  sizeax[2] = local_n0;
  sizeax[1] = Ny;
  global_x_size = Nx;

  if (typeid(T) == typeid(double)) {
    sizeax[0] = Nx;
    arr = (T*) fftw_alloc_real(2*alloc_local);
    spacer = 2*(Nx/2+1);    
    size = spacer*alloc_local;
  } else if (typeid(T) == typeid(std::complex<double>)) {
    sizeax[0] = Nx/2+1;
    arr = (T*) fftw_alloc_complex(alloc_local);
    spacer=Nx/2+1;
    size = alloc_local;
  } else
    throw std::runtime_error("array3D can only have type double, "
			     "or std::complex<double>.");
  
  array_name = name;
  
};



template <typename T>
array3D<T>::~array3D() {
  fftw_free(arr);
}



template <typename T>
T& array3D<T>::operator()(ptrdiff_t nx,ptrdiff_t ny, ptrdiff_t nz)
/* read/write access array elements via (nx,ny,nz). */
{

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];
}



template <typename T>
T array3D<T>::operator()(ptrdiff_t nx,ptrdiff_t ny, ptrdiff_t nz) const
/* read-only access (but don't change) array elements via (nx,ny,nz). */
{
  return arr[nx + (nz*sizeax[1] + ny ) * spacer];

}




template <typename T>
T& array3D<T>::operator()(ptrdiff_t flat)
/* read/write access array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = (flat/sizeax[0]) % sizeax[1];
  int nz = (flat/sizeax[0])/sizeax[1];

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];
}



template <typename T>
T array3D<T>::operator()(ptrdiff_t flat) const
/* read-only access (but don't change) array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = (flat/sizeax[0]) % sizeax[1];
  int nz = (flat/sizeax[0])/sizeax[1];

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];

}



template <typename T>
array3D<T>& array3D<T>::operator=(array3D<T> other)
{
  swap(*this,other);
  
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator=(T other)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny) * spacer] = other;
      }
    }
  }
  return *this;
}




template <typename T>
void array3D<T>::reverseFlat(int gridindex, int &nx, int &ny, int &nz) const
{

  nx = gridindex % sizeax[0];
  ny = (gridindex / sizeax[0]) % sizeax[1];
  nz = (gridindex / sizeax[0]) / sizeax[1];

}



template <typename T>
array3D<T>& array3D<T>::operator*=(T rhs)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] *= rhs;
      }
    }
  }
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator/=(T rhs)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] /= rhs;
      }
    }
  }
  return *this;
}

template <typename T>
array3D<T>& array3D<T>::operator+=(T rhs)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] += rhs;
      }
    }
  }
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator-=(T rhs)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] -= rhs;
      }
    }
  }
  return *this;
}

template <typename T>
array3D<T>& array3D<T>::operator*=(const array3D<T>& rhs)
{
  if (Nz() != rhs.Nz() || Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise multiplication");
    throw std::runtime_error(errmsg);
  }  
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] *= rhs(nx,ny,nz);
      }
    }
  }
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator/=(const array3D<T>& rhs)
{

  if (Nz() != rhs.Nz() || Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise division");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] /= rhs(nx,ny,nz);
      }
    }
  }
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator+=(const array3D<T>& rhs)
{

  if (Nz() != rhs.Nz() || Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise addition");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] += rhs(nx,ny,nz);
      }
    }
  }
  return *this;
}


template <typename T>
array3D<T>& array3D<T>::operator-=(const array3D<T>& rhs)
{

  if (Nz() != rhs.Nz() || Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise subtraction");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] -= rhs(nx,ny,nz);
      }
    }
  }
  return *this;
}


template <typename T>
void array3D<T>::write_to_binary()
{

  int recvid, sendid;

  if (!fftw_recv)
    fftw_recv = std::make_unique<array3D<T>>(world,array_name+std::string("_neighborplane"),
					     global_x_size,sizeax[1],nprocs);
      

}


template <typename T>
std::string array3D<T>::operation_err_msg(const std::string & othername,
					  const std::string & operation)
{
    std::string errmsg;
    errmsg = operation + std::string(" of fftwArrs failed: ");
    errmsg += array_name + std::string(" and ") + othername;
    errmsg += std::string(" cannot be broadcast (sizes don't match).");

    return errmsg;

}



template class fftwArr::array3D<double>;
template class fftwArr::array3D<std::complex<double>>;
/*
template <typename T>
void array3D<T>::abs(array3D& modulus) const
{


    
  if (Nz() != modulus.Nz() || Ny() != modulus.Ny() || Nx() != modulus.Nx())
      throw std::runtime_error("Cannot take abs of array3D (wrong output shape).");



  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	modulus(nx,ny,nz) = std::abs(arr[nx + (nz*sizeax[1] + ny ) * spacer]);
      }
    }
  }
  return;

}


template <typename T>
void array3D<T>::modSq(array3D& modulus) const
{


  if (Nz() != modulus.Nz() || Ny() != modulus.Ny() || Nx() != modulus.Nx())  
    throw std::runtime_error("Cannot take abs of array3D (wrong output shape).");



  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	modulus(nx,ny,nz) = std::abs(arr[nx + (nz*sizeax[1] + ny ) * spacer])
	  *std::abs(arr[nx + (nz*sizeax[1] + ny ) * spacer]);
      }
    }
  }
  return;

}



template <typename T>
void array3D<T>::running_modSq(array3D& modulus) const
{

  if (Nz() != modulus.Nz() || Ny() != modulus.Ny() || Nx() != modulus.Nx())  
    throw std::runtime_error("Cannot take abs of array3D (wrong output shape).");


  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	modulus(nx,ny,nz) += std::abs(arr[nx + (nz*sizeax[1] + ny ) * spacer])
	  *std::abs(arr[nx + (nz*sizeax[1] + ny ) * spacer]);
      }
    }
  }
  return;

}

*/
