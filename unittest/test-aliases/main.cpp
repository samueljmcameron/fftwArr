#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
#include <memory>

#include "fftw_arr/arrays.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

int main()
{
  
  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  int Nx = 20;
  int Ny = 40;
  int Nz = 30;

  fftw_mpi_init();


  fftwArr::r2c_2D phi(MPI_COMM_WORLD,"phi",Nx,Ny);
  fftwArr::c2r_2D ft_phi(MPI_COMM_WORLD,"phi",Nx,Ny);
  fftwArr::c2c_2D complex_phi(MPI_COMM_WORLD,"phi",Nx,Ny);


  fftwArr::r2c_3D theta(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);
  fftwArr::c2r_3D ft_theta(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);
  fftwArr::c2c_3D complex_theta(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);
  
  fftw_mpi_cleanup();


  
  if (nprocs > 1)
    std::cout << "WARNING: ostream output will be out of order in general for nprocs > 1." << std::endl;


  ierr = MPI_Finalize();

  return 0;
}
