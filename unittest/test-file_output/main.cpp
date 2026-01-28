#include <mpi.h>
#include <iostream>
#include <complex>
#include <memory>
#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm ,int );

int main()
{
  
  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);


  fftw_mpi_init();

  for (int dim = 2; dim <= 3; dim++) {
    test_function<fftwArr::Transform::R2C,double>(world,dim);

    test_function<fftwArr::Transform::C2R,std::complex<double>>(world,dim);

    test_function<fftwArr::Transform::C2C,std::complex<double>>(world,dim);

  }

  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

template < enum fftwArr::Transform rOc,typename T>
void test_function(MPI_Comm world,int dim)
{

  int me;
  MPI_Comm_rank(world,&me);

  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());


  /* First block of the code is to open a file which the array data
     will be written to. */
    
  std::string filename = std::string("generated_files/");
  std::string arrname = fftwArrTestingUtils::fftwArrName(dtype,rOc,dim);
  filename = filename + arrname + "_p" + std::to_string(me) +  std::string(".out");

  std::fstream writefile;
  
  writefile.open(filename,
		 std::fstream::binary | std::fstream::out);


  int local_flag,global_flag;
  std::string broken_file = "";
  local_flag = 0;
  
  if (!writefile) {
    local_flag = 1;
    broken_file = "Failed to open (write-only) " + filename;
  }

  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		world);

  if (global_flag > 0)
    throw std::runtime_error(broken_file);



  /* Second block of the code is to write data to the appropriate array, which
     in this case is phi_2d or phi_3d .*/
  
  std::unique_ptr<fftwArr::array2D<rOc,T>> phi_2d;
  std::unique_ptr<fftwArr::array2D<rOc,T>> theta_2d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> phi_3d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> theta_3d;  


  if (dim == 2) {
    int Nx = 3;
    int Ny = 13;
    
    
    phi_2d = std::make_unique<fftwArr::array2D<rOc,T>>(world,"phi_2d",Nx,Ny);
    theta_2d = std::make_unique<fftwArr::array2D<rOc,T>>(world,"theta_2d",Nx,Ny);


    for (int ny = 0; ny < phi_2d->Ny(); ny++)
      for (int nx = 0; nx < phi_2d->Nx(); nx++)
	(*phi_2d)(nx,ny) = nx+ny + 0.4;
    
    
    phi_2d->write_to_binary(writefile);
    

  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 2;
    int Nz = 10;
    
    
    
    phi_3d = std::make_unique<fftwArr::array3D<rOc,T>>(world,"phi_3d",Nx,Ny,Nz);
    theta_3d = std::make_unique<fftwArr::array3D<rOc,T>>(world,"theta_3d",Nx,Ny,Nz);
    
    for (int nz = 0; nz < phi_3d->Nz(); nz++)
      for (int ny = 0; ny < phi_3d->Ny(); ny++)
	for (int nx = 0; nx < phi_3d->Nx(); nx++)
	  (*phi_3d)(nx,ny,nz) = nx+ny+nz + 0.4;
    
    
    phi_3d->write_to_binary(writefile);
  }

  writefile.close();



  /* Third block of the code is to open a read only file .*/

    
  std::fstream readfile;
  
  readfile.open(filename,
		std::fstream::binary | std::fstream::in);
  
  
  if (!readfile) {
    local_flag = 1;
    broken_file = "Failed to open (read-only) " + filename;
  }
  
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		world);
  
  if (global_flag > 0)
    throw std::runtime_error(broken_file);


  /* Fourth block of the code is to read data to the appropriate array, which
     in this case is theta_2d or theta_3d .*/

  
  if (dim == 2) {
    
    theta_2d->read_from_binary(readfile);
    
    for (int ny = 0; ny < theta_2d->Ny(); ny ++ )
      for (int nx = 0; nx < theta_2d->Nx(); nx++)
	if (std::abs((*theta_2d)(nx,ny)-(*phi_2d)(nx,ny)) > 1e-3)
	  local_flag = 1;
  } else if (dim == 3) {

    theta_3d->read_from_binary(readfile);
    
    for (int nz = 0; nz < theta_3d->Nz(); nz ++ )
      for (int ny = 0; ny < theta_3d->Ny(); ny ++ )
	for (int nx = 0; nx < theta_3d->Nx(); nx++)
	  if (std::abs((*theta_3d)(nx,ny,nz)-(*phi_3d)(nx,ny,nz)) > 1e-3)
	    local_flag = 1;
  }
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		world);
  
  if (global_flag > 0)
    throw std::runtime_error("MISMATCHED I/O");
  
  
  if (me == 0)
    std::cout << fftwArrTestingUtils::SuccessMessage(dtype,rOc,dim) << std::endl;
  
}
