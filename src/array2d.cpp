#include "array2d.hpp"

using namespace fftwArr;

template < enum Transform rOc,typename T>
array2D<rOc,T>::array2D()
/*
  Default constructor (does nothing).
*/
{
  arr=nullptr;
};


template < enum Transform rOc,typename T>
array2D<rOc,T>::array2D(const MPI_Comm &comm,std::string name,
			 ptrdiff_t Nx, ptrdiff_t Ny)
/*
  Constructor for a 2D array with axis sizes (Nx,Ny) (the x dimension
  varies the quickest). The array is not contiguous in memory for different
  y indices. The array is split across the y index, and is in
  Fortran (column-major) order to be compatible with vtk (so that the x
  index varies the quickest).

  The structure (once parallelised will be):

  x
  |
  |
  |
   ------ y
  
      ______     ______     ______            ______
     |      |   |      |   |      |          |      |   
     |      |   |      |   |      |          |      |   
     |      |   |      |   |      |     ...  |      |   
     |      |   |      |   |      |          |      |   
 Nx  |      |   |      |   |      |          |      |   
     |      |   |      |   |      |          |      |   
     |      |   |      |   |      |          |      |
     |      |   |      |   |      |          |      |
     |______|   |______|   |______|          |______|

       Ny_1       Ny_1       Ny_2       ...    Ny_p



------------------------------------------------------------
  
  GLOBALLY, the array (in the complex to complex case) has
  size

      Nx x Ny

  which is contigous in memory. The same ordering of how
  indices line up with q space as mentioned below applies.


------------------------------------------------------------

  GLOBALLY, the real array (in the complex to real case) has
  size

      Nx x Ny

  (though the space held in memory replaces Nx with 2(Nx/2+1)
  by design of FFTW3 transforms).

  
------------------------------------------------------------
  GLOBALLY, the complex array (in the complex to real case) has
  size

      (Nx/2+1) x Ny 

  and the array storage due to FFTW3 is such that the indices

      0,1,...,Nx/2

  represent qx >= 0, and similarly for y. However,
  for y there are also indices

      Ny/2+1,Ny/2+2,...Ny-1


  and these indices map to the qy values

      ny -> 2*pi*(Ny-ny)/Ly
      
  for ny > Ny/2. Locally, the local_ny index is determined by

      local_ny = local_0_start - ny

  where ny is the global ny index.


------------------------------------------------------------
  Note that FFTW3 library also has an option where the
  complex array has the final two indices swapped relative
  to the real array which speeds up computations (passing
  FFTW_TRANSFORM flags). In this case one would want to
  ensure that they swap the order in which they pass the Nx
  and Ny indices in creating a std::complex<double>
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
*/
{


  ptrdiff_t local_n0;
  
  world = comm;

  

  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  // pass axes along so that z varies most slowly, then y, then x

  if (rOc == Transform::C2C) {


    alloc_local = fftw_mpi_local_size_2d(Ny , Nx,
					 world,&local_n0,&local_0_start);
    
    sizeax[1] = local_n0;
    sizeax[0] = Nx;
    global_x_size = Nx;

    arr = (T*) fftw_alloc_complex(alloc_local);
    spacer=Nx;
    size = alloc_local;

    
  } else {
    alloc_local = fftw_mpi_local_size_2d(Ny , Nx/2 + 1,
					 world,&local_n0,&local_0_start);
    
    sizeax[1] = local_n0;
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
      throw std::runtime_error("array2D can only have type double, "
			       "or std::complex<double>.");
    
  }

  int processor_used = 1;
  
  if (sizeax[1] == 0)
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
array2D<rOc,T>::array2D(const array2D<rOc,T> & base,std::string name)
  : alloc_local(base.alloc_local),local_0_start(base.local_0_start),
    size(base.size),array_name(base.array_name), spacer(base.spacer),
    global_x_size(base.global_x_size),nprocs(base.nprocs),me(base.me),world(base.world)
/*
  Copy array, but if name (other than "") is provided then only make an
  array of the same size with the new name, but don't copy the elements in the
  array.

  Parameters
  ----------
  base : array2D
      The array to either copy from or set size structures from
  name : std::string (optional)
      The name of the new array, default is "" which makes this a true copy constructor.
      
*/


{


  sizeax[0] = base.sizeax[0];
  sizeax[1] = base.sizeax[1];


  

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
      throw std::runtime_error("array2D can only have type double, "
			       "or std::complex<double>.");
  }
    
  if (name != "") array_name = name;
  else std::copy(base.arr, base.arr + tmpsize,arr);

  
}




template < enum Transform rOc,typename T>
array2D<rOc,T>::~array2D() {
  fftw_free(arr);
}



template < enum Transform rOc,typename T>
T& array2D<rOc,T>::operator()(ptrdiff_t nx,ptrdiff_t ny)
/* read/write access array elements via (nx,ny). */
{

  return arr[nx +  ny * spacer];
}



template < enum Transform rOc,typename T>
T array2D<rOc,T>::operator()(ptrdiff_t nx,ptrdiff_t ny) const
/* read-only access (but don't change) array elements via (nx,ny). */
{
  return arr[nx +  ny * spacer];

}




template < enum Transform rOc,typename T>
T& array2D<rOc,T>::operator()(ptrdiff_t flat)
/* read/write access array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = flat/sizeax[0];

  return arr[nx +  ny * spacer];
}



template < enum Transform rOc,typename T>
T array2D<rOc,T>::operator()(ptrdiff_t flat) const
/* read-only access (but don't change) array elements via flattened array. */
{

  int nx = flat % sizeax[0];
  int ny = flat/sizeax[0];

  return arr[nx +  ny * spacer];

}



template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator=(array2D<rOc,T> other)
{
  swap(*this,other);
  
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator=(T other)
{
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx + ny * spacer] = other;
    }
  }
  
  return *this;
}




template < enum Transform rOc,typename T>
void array2D<rOc,T>::reverseFlat(int gridindex, int &nx, int &ny) const
{

  nx = gridindex % sizeax[0];
  ny = gridindex / sizeax[0];


}



template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator*=(T rhs)
{
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] *= rhs;
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator/=(T rhs)
{
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] /= rhs;
    }
  }
  return *this;
}

template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator+=(T rhs)
{
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] += rhs;
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator-=(T rhs)
{
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] -= rhs;
    }
  }
  return *this;
}

template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator*=(const array2D<rOc,T>& rhs)
{
  if (Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise multiplication");
    throw std::runtime_error(errmsg);
  }  
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] *= rhs(nx,ny);
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator/=(const array2D<rOc,T>& rhs)
{

  if (Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise division");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] /= rhs(nx,ny);
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator+=(const array2D<rOc,T>& rhs)
{

  if (Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise addition");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] += rhs(nx,ny);
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
array2D<rOc,T>& array2D<rOc,T>::operator-=(const array2D<rOc,T>& rhs)
{

  if (Ny() != rhs.Ny() || Nx() != rhs.Nx()) {
    std::string errmsg
      = operation_err_msg(rhs.get_name(),"Element-wise subtraction");
    throw std::runtime_error(errmsg.c_str());
  }
  
  for (int ny = 0; ny < sizeax[1]; ny++) {
    for (int nx = 0; nx < sizeax[0]; nx++) {
      arr[nx +  ny * spacer] -= rhs(nx,ny);
    }
  }
  return *this;
}


template < enum Transform rOc,typename T>
void array2D<rOc,T>::write_to_binary(std::fstream &myfile,
				 const bool overlap)
/* write the current processor's array data to a binary file.
   If overlap = true, then the file also shares one x line of
   data with each of its neighboring processors (with the exception
   of the first and last processor, who only share with the second
   and second last, respectively).
   This format is compatible with vtk files. and the data is prefixed
   with the bytelength of the array:

   bytelength arr(0,0) arr(1,0) ...
   
*/
{

  int recvid, sendid;

  if (!fftw_recv && overlap)
    fftw_recv
      = std::make_unique<array2D<rOc,T>>(world,array_name
				     +std::string("_neighborplane"),
				     global_x_size,nprocs);



  int pad;
  if (!overlap || nprocs == 1)
    pad = 0;
  else if (me == 0 || me == nprocs-1)
    pad = 1;
  else
    pad = 2;
  
  unsigned int bytelength;
  bytelength = sizeax[0]*(sizeax[1]+pad)*sizeof(T);


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



    MPI_Sendrecv(&(*this)(0,sizeax[1]-1),this->xsize(),
		 MPI_DOUBLE,sendid,0,
		 fftw_recv->data(),fftw_recv->xsize(),
		 MPI_DOUBLE,recvid,0,world,MPI_STATUS_IGNORE);

    if (me != 0)

      myfile.write((char*)&(*fftw_recv)(0,0),
		   sizeof(T)*fftw_recv->Nx());

  }

  for (int ny = 0; ny < sizeax[1]; ny++)
    myfile.write((char*)&(*this)(0,ny),sizeof(T)*sizeax[0]);


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
    
    
    MPI_Sendrecv(&(*this)(0,0),this->xsize(),
		 MPI_DOUBLE,sendid,0,
		 fftw_recv->data(),fftw_recv->xsize(),
		 MPI_DOUBLE,recvid,0,world,MPI_STATUS_IGNORE);

  
  if (me != nprocs-1)
    myfile.write((char*)&(*fftw_recv)(0,0),
		 sizeof(T)*fftw_recv->Nx());  

  }

  return;

}



template < enum Transform rOc,typename T>
void array2D<rOc,T>::read_from_binary(std::fstream &myfile,
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


  unsigned int expected_size = (sizeax[1]+factor)*sizeax[0]*sizeof(T);
  if (bytelength != expected_size) {
    throw std::runtime_error("size mismatch between "
			     + array_name + "and binary data in file."
			     + " Expected size vs received size is "
			     + std::to_string(expected_size) + " vs "
			     + std::to_string(bytelength));
  }
  
  // if not processor zero, then we need to read the extra bit at the front of the data file
  for (int ny = 0; ny < front_offset; ny++) {
    myfile.ignore(sizeof(T)*sizeax[0]);
  }


  // since real fftw arrays aren't contiguous, need to read each row separately.
  for (int ny = 0; ny < sizeax[1]; ny++) {
    myfile.read((char*)&(*this)(0,ny),sizeof(T)*sizeax[0]);
  }

  // if not the last processor, then we need to read the extra bit at the end of the data file
  for (int ny = 0; ny < back_offset; ny++) {
    myfile.ignore(sizeof(T)*sizeax[0]);
  }

  return;

}  



template < enum Transform rOc,typename T>
std::string array2D<rOc,T>::operation_err_msg(const std::string & othername,
					  const std::string & operation)
{
    std::string errmsg;
    errmsg = operation + std::string(" of fftwArrs failed: ");
    errmsg += array_name + std::string(" and ") + othername;
    errmsg += std::string(" cannot be broadcast (sizes don't match).");

    return errmsg;

}



template class fftwArr::array2D<fftwArr::Transform::R2C,double>;
template class fftwArr::array2D<fftwArr::Transform::C2R,std::complex<double>>;
template class fftwArr::array2D<fftwArr::Transform::C2C,std::complex<double>>;
