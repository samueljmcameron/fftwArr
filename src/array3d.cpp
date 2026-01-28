#include "array3d.hpp"

using namespace fftwArr;

template < enum Transform rOc,typename T>
array3D<rOc,T>::array3D()
/*
  Default constructor (does nothing).
*/
{
  arr=nullptr;
};


template < enum Transform rOc,typename T>
array3D<rOc,T>::array3D(const MPI_Comm &comm,std::string name,
			ptrdiff_t Nx, ptrdiff_t Ny, ptrdiff_t Nz)
/*
  Constructor for a 3D array with axis sizes (Nx,Ny,Nz) (the x dimension
  varies the quickest). The array is not contiguous in memory for different
  y and z indices. The array is split across the z index, and is in
  Fortran (column-major) order to be compatible with vtk (so that the x
  index varies the quickest).

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



------------------------------------------------------------
  
  GLOBALLY, the array (in the complex to complex case) has
  size

      Nx x Ny x Nz

  which is contigous in memory. The same ordering of how
  indices line up with q space as mentioned below applies.


------------------------------------------------------------
  
  GLOBALLY, the real array (in the complex to real case) has
  size

      Nx x Ny x Nz

  (though the space held in memory replaces Nx with 2(Nx/2+1)
  by design of FFTW3 transforms).


  
------------------------------------------------------------
  GLOBALLY, the complex array (in the complex to real case)
  has size

      (Nx/2+1) x Ny x Nz

  and the array storage due to FFTW3 is such that indices

      0,1,...,Nx/2

  represent qx >= 0, and similarly for y and z. However,
  for y and z there are also indices

      Ny/2+1,Ny/2+2,...Ny-1
      Nz/2+1,Nz/2+2,...Nz-1

  and these indices map to the qy and qz values

      ny -> 2*pi*(Ny-ny)/Ly
      nz -> 2*pi*(Nz-nz)/Ly
      
  for ny > Ny/2 and nz > Nz/2. Locally, the local_nz
  index is determined by

      local_nz = local_0_start - nz

  where nz is the global nz index.



------------------------------------------------------------
  Note that FFTW3 library also has an option where the
  complex array has the final two indices swapped relative
  to the real array which speeds up computations (passing
  FFTW_TRANSFORM flags). In this case one would want to
  ensure that they swap the order in which they pass the Ny
  and Nz indices in creating a std::complex<double>
  instance of this class.

       
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


  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);
  
  // pass axes along so that z varies most slowly, then y, then x


  global_x_size = Nx;
  global_z_size = Nz;
  
  if (rOc == Transform::C2C) {
    

    alloc_local = fftw_mpi_local_size_3d(Nz, Ny , Nx,
					 world,&local_n0,&local_0_start);
    
    sizeax[2] = local_n0;
    sizeax[1] = Ny;
    sizeax[0]= Nx;

    arr = (T*) fftw_alloc_complex(alloc_local);
    spacer=Nx;
    size = alloc_local;
    

  } else {
    alloc_local = fftw_mpi_local_size_3d(Nz, Ny , Nx/2 + 1,
					 world,&local_n0,&local_0_start);
  
    sizeax[2] = local_n0;
    sizeax[1] = Ny;

    
        
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
    
  }

  int processor_used = 1;
  if (sizeax[2] == 0)
    processor_used = 0;
  
  int global_procs;
  
  MPI_Allreduce(&processor_used, &global_procs, 1, MPI_INT, MPI_SUM, world);
  
  if (global_procs != nprocs)
    throw std::runtime_error("Only " + std::to_string(global_procs)
			     + " out of " + std::to_string(nprocs)
			     + " processors have fftwArr data stored.");
  
  
  
  array_name = name;
  
  // the next line sets all elements of the array equal to zero
  *this = 0;
};





template < enum Transform rOc,typename T>
array3D<rOc,T>::array3D(const array3D<rOc,T> & base,std::string name)
  : alloc_local(base.alloc_local),local_0_start(base.local_0_start),
    size(base.size),array_name(base.array_name), spacer(base.spacer),
    global_x_size(base.global_x_size),global_z_size(base.global_z_size),
    nprocs(base.nprocs),me(base.me),world(base.world)
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

  if (rOc == Transform::C2C) {
    arr = (T*) fftw_alloc_complex(alloc_local);
    tmpsize = alloc_local;

  } else {

    
    if (typeid(T) == typeid(double)) {
      arr = (T*) fftw_alloc_real(2*alloc_local);
      
      tmpsize = 2*alloc_local;
    } else if (typeid(T) == typeid(std::complex<double>)) {
      arr = (T*) fftw_alloc_complex(alloc_local);
      tmpsize = alloc_local;
      
    } else
      throw std::runtime_error("array3D can only have type double, "
			       "or std::complex<double>.");
  }
    
  if (name != "") array_name = name;
  else std::copy(base.arr, base.arr + tmpsize,arr);

  
}


template < enum Transform rOc,typename T>
array3D<rOc,T>::~array3D() {
  fftw_free(arr);
}



template < enum Transform rOc,typename T>
T& array3D<rOc,T>::operator()(ptrdiff_t nx,ptrdiff_t ny, ptrdiff_t nz)
/* read/write access array elements via (nx,ny,nz). */
{

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];
}



template < enum Transform rOc,typename T>
T array3D<rOc,T>::operator()(ptrdiff_t nx,ptrdiff_t ny, ptrdiff_t nz) const
/* read-only access (but don't change) array elements via (nx,ny,nz). */
{
  return arr[nx + (nz*sizeax[1] + ny ) * spacer];

}




template < enum Transform rOc,typename T>
T& array3D<rOc,T>::operator()(ptrdiff_t flat)
/* read/write access array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = (flat/sizeax[0]) % sizeax[1];
  int nz = (flat/sizeax[0])/sizeax[1];

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];
}



template < enum Transform rOc,typename T>
T array3D<rOc,T>::operator()(ptrdiff_t flat) const
/* read-only access (but don't change) array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = (flat/sizeax[0]) % sizeax[1];
  int nz = (flat/sizeax[0])/sizeax[1];

  return arr[nx + (nz*sizeax[1] + ny ) * spacer];

}



template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator=(array3D<rOc,T> other)
{
  swap(*this,other);
  
  return *this;
}


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator=(T other)
{
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      for (int nx = 0; nx < sizeax[0]; nx++) {
	arr[nx + (nz*sizeax[1] + ny ) * spacer] = other;
      }
    }
  }
  return *this;
}




template < enum Transform rOc,typename T>
void array3D<rOc,T>::reverseFlat(int gridindex, int &nx, int &ny, int &nz) const
{

  nx = gridindex % sizeax[0];
  ny = (gridindex / sizeax[0]) % sizeax[1];
  nz = (gridindex / sizeax[0]) / sizeax[1];

}



template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator*=(T rhs)
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


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator/=(T rhs)
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

template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator+=(T rhs)
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


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator-=(T rhs)
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

template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator*=(const array3D<rOc,T>& rhs)
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


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator/=(const array3D<rOc,T>& rhs)
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


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator+=(const array3D<rOc,T>& rhs)
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


template < enum Transform rOc,typename T>
array3D<rOc,T>& array3D<rOc,T>::operator-=(const array3D<rOc,T>& rhs)
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


template < enum Transform rOc,typename T>
void array3D<rOc,T>::write_to_binary(std::fstream &myfile,
				 const bool overlap)
/* write the current processor's array data to a binary file.
   If overlap = true, then the file also shares one xy plane of
   data with each of its neighboring processors (with the exception
   of the first and last processor, who only share with the second
   and second last, respectively).
   This format is compatible with vtk files. and the data is prefixed
   with the bytelength of the array:

   bytelength arr(0,0,0) arr(1,0,0) ...
   
*/
{

  int recvid, sendid;

  if (!fftw_recv && overlap)
    fftw_recv
      = std::make_unique<array3D<rOc,T>>(world,array_name
				     +std::string("_neighborplane"),
				     global_x_size,sizeax[1],nprocs);



  int pad;
  if (!overlap || nprocs == 1)
    pad = 0;
  else if (me == 0 || me == nprocs-1)
    pad = 1;
  else
    pad = 2;
  
  unsigned int bytelength;
  bytelength = sizeax[0]*sizeax[1]*(sizeax[2]+pad)*sizeof(T);


  myfile.write((char*)&bytelength,sizeof(bytelength));



  if (nprocs > 1 && overlap) {

    // write first overlap (receive from left, send to right)
    
    if (me == 0) {
      recvid = nprocs-1;
      sendid = me+1;
      
    } else if (me == nprocs-1) {
      recvid = me-1;
      sendid = 0;
      
    } else {
      recvid = me -1;
      sendid = me+1;
    }



    MPI_Sendrecv(&(*this)(0,0,sizeax[2]-1),this->xysize(),
		 MPI_DOUBLE,sendid,0,
		 fftw_recv->data(),fftw_recv->xysize(),
		 MPI_DOUBLE,recvid,0,world,MPI_STATUS_IGNORE);

    if (me != 0)

      for (int ny = 0; ny < fftw_recv->Ny(); ny++)
	myfile.write((char*)&(*fftw_recv)(0,ny,0),
		     sizeof(T)*fftw_recv->Nx());

  }

  for (int nz = 0; nz < sizeax[2]; nz++) 
    for (int ny = 0; ny < sizeax[1]; ny++)
      myfile.write((char*)&(*this)(0,ny,nz),sizeof(T)*sizeax[0]);


  if (nprocs > 1 && overlap) {

    // now send to left/recv from right
    if (me == 0) {
      recvid = me+1;
      sendid = nprocs-1;
      
    } else if (me == nprocs-1) {
      recvid = 0;
      sendid = me-1;
      
    } else {
      recvid = me+1;
      sendid = me -1;
    }
    
    
    MPI_Sendrecv(&(*this)(0,0,0),this->xysize(),
		 MPI_DOUBLE,sendid,0,
		 fftw_recv->data(),fftw_recv->xysize(),
		 MPI_DOUBLE,recvid,0,world,MPI_STATUS_IGNORE);

  
  if (me != nprocs-1)
    for (int ny = 0; ny < fftw_recv->Ny(); ny++)
      myfile.write((char*)&(*fftw_recv)(0,ny,0),
		   sizeof(T)*fftw_recv->Nx());  

  }

  return;

}



template < enum Transform rOc,typename T>
void array3D<rOc,T>::read_from_binary(std::fstream &myfile,
				  const bool overlap)
{
  unsigned int bytelength;

  myfile.read((char*)&bytelength,sizeof(bytelength));



  int factor = 2;

  int front_offset = 1;
  int back_offset = 1;


  if (!overlap || nprocs == 1) {
    factor = front_offset = back_offset = 0;
  } else if (me == 0) {
    factor -= 1;
    front_offset = 0;
  } else if (me == nprocs-1) {
    factor -= 1;
    back_offset=0;
  }


  unsigned int expected_size = (sizeax[2]+factor)*sizeax[1]*sizeax[0]*sizeof(T);
  if (bytelength != expected_size) {
    throw std::runtime_error("size mismatch between "
			     + array_name + "and binary data in file."
			     + " Expected size vs received size is "
			     + std::to_string(expected_size) + " vs "
			     + std::to_string(bytelength));
  }
  
  // if not processor zero, then we need to read the extra bit at the front of the data file
  for (int nz = 0; nz < front_offset; nz++) {
    myfile.ignore(sizeof(T)*sizeax[0]*sizeax[1]);
  }


  // since real fftw arrays aren't contiguous, need to read each row separately.
  for (int nz = 0; nz < sizeax[2]; nz++) {
    for (int ny = 0; ny < sizeax[1]; ny++) {
      myfile.read((char*)&(*this)(0,ny,nz),sizeof(T)*sizeax[0]);

    }
  }

  // if not the last processor, then we need to read the extra bit at the end of the data file
  for (int nz = 0; nz < back_offset; nz++) {
    myfile.ignore(sizeof(T)*sizeax[0]*sizeax[1]);
  }

  return;

}  



template < enum Transform rOc,typename T>
std::string array3D<rOc,T>::operation_err_msg(const std::string & othername,
					  const std::string & operation)
{
    std::string errmsg;
    errmsg = operation + std::string(" of fftwArrs failed: ");
    errmsg += array_name + std::string(" and ") + othername;
    errmsg += std::string(" cannot be broadcast (sizes don't match).");

    return errmsg;

}



template class fftwArr::array3D<fftwArr::Transform::R2C,double>;
template class fftwArr::array3D<fftwArr::Transform::C2R,std::complex<double>>;
template class fftwArr::array3D<fftwArr::Transform::C2C,std::complex<double>>;
/*
template < enum Transform rOc,typename T>
void array3D<rOc,T>::abs(array3D& modulus) const
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


template < enum Transform rOc,typename T>
void array3D<rOc,T>::modSq(array3D& modulus) const
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



template < enum Transform rOc,typename T>
void array3D<rOc,T>::running_modSq(array3D& modulus) const
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
