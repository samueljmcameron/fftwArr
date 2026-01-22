#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
#include "fftw_arr/array3d.hpp"

template <typename T>
void test_function(MPI_Comm world);

int main()
{
  
  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  
  fftw_mpi_init();


  // see test-ostream for explanation of the next
  // three lines
  std::tuple<double,std::complex<double>> types;

  std::apply([=](auto... args){((test_function<decltype(args)>(world)),...);},types);  

  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}


template <typename T>
void test_function(MPI_Comm world)
{
  int Nx = 20;
  int Ny = 2;
  int Nz = 3;

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::array3D<T> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny,Nz);
  fftwArr::array3D<T> phi_2;
  
  phi_2 = fftwArr::array3D<T>(MPI_COMM_WORLD,"phi_2",Nx,Ny,Nz);

  phi_1 += 2.0;
  phi_1 /= 2.0;
  phi_1 *= 3.0;
  phi_1 -= 2.0;
  
  phi_2 += 3.0;

  phi_1 += phi_2;

  if (phi_1.get_me() == 0)
    std::cout << "\nTesting for fftwArr of type " << typeid(T).name()
	      << " was successful.\n" << std::endl;
  
}
