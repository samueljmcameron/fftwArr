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
  int Ny = 7;
  int Nz = 13;

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::array3D<T> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny,Nz);
  fftwArr::array3D<T> phi_2;
  
  phi_2 = fftwArr::array3D<T>(MPI_COMM_WORLD,"phi_2",Nx/2,Ny,Nz);

  phi_2 = 1.0;

  
  try {
    
    phi_1 += phi_2;
    
  } catch (const std::runtime_error& add_except) {
    
    std::cerr << add_except.what() << std::endl;
    
    try {
      
      phi_1 -= phi_2;
      
    } catch (const std::runtime_error& sub_except) {
      
      std::cerr << sub_except.what() << std::endl;
      
      try {
	
	phi_1 *= phi_2;
	
      } catch (const std::runtime_error& times_except) {
	
	std::cerr << times_except.what() << std::endl;
	
	try {
	  
	  phi_1 /= phi_2;
	  
	} catch (const std::runtime_error& div_except) {
	  
	  std::cerr << div_except.what() << std::endl;
	  
	}
      }
    }
  }

  if (phi_1.get_me() == 0)
    std::cout << "\nTesting for fftwArr of type " << typeid(T).name()
	      << " was successful.\n" << std::endl;
  
}
