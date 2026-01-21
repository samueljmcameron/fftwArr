#include <mpi.h>
#include <iostream>
#include "fftw_arr/array3d.hpp"


int main()
{
  
  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  
  fftw_mpi_init();

  int Nx = 20;
  int Ny = 2;
  int Nz = 3;
  double L = 2*M_PI;

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::array3D<double> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny,Nz);
  fftwArr::array3D<double> phi_2;

  phi_2 = fftwArr::array3D<double>(MPI_COMM_WORLD,"phi_2",Nx,Ny,Nz);


  for (int nx = 0; nx < phi_2.Nx(); nx++)
    for (int ny = 0; ny < phi_2.Ny(); ny++)
      for (int nz = 0; nz < phi_2.Nz(); nz++)
  	phi_2(nx,ny,nz) = 3.0;



  /*
  int new_nprocs = phi_1.get_nprocs();
  int new_me = phi_1.get_me();
  MPI_Comm new_world = phi_1.get_world();

  if (new_me == 0)
    std::cout << "Proc\tIndex\t" << phi_1.get_name() << std::endl;
  
  for (int p = 0; p < new_nprocs; p++) {
    if (p == new_me) {
      
      for (int i = 0; i < phi_1.Nx(); i++)
	for (int j = 0; j < phi_1.Ny(); j++)
	  for (int k = 0; k < phi_1.Nz(); k++)
	    std::cout << p << "\t" << "(" << i << "," << j << ","
		      << k << ")\t" << phi_1(i,j,k) << std::endl;
    }
    MPI_Barrier(new_world);
    
  }
  */
  
  
  std::cout << phi_1 << std::endl;
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

