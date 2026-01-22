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

  // what follows is a hack to iterate over different types.

  // first, define a tuple with all the types you want to iterate over
  std::tuple<double,std::complex<double>> types;

  // std::apply invokes the lambda function with elements of type as arguments
  
  // NOTE: can't use && as that just provides a reference to the tuple elements,
  // which would make e.g. decltype(std::get<0>(types)) return &double instead of
  // double
  std::apply([=](auto... args){((test_function<decltype(args)>(world)),...);},types);  


  //test_function<double>(world);

  //test_function<std::complex<double>>(world);
  
  fftw_mpi_cleanup();


  
  if (nprocs > 1)
    std::cout << "WARNING: ostream output will be out of order in general for nprocs > 1." << std::endl;


  ierr = MPI_Finalize();

  return 0;
}

template <typename T>
void test_function(MPI_Comm world)
{
  int Nx = 10;
  int Ny = 2;
  int Nz = 3;

  
  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::array3D<T> phi_1(world,"phi_1",Nx,Ny,Nz);


  for (int nx = 0; nx < phi_1.Nx(); nx++)
    for (int ny = 0; ny < phi_1.Ny(); ny++)
      for (int nz = 0; nz < phi_1.Nz(); nz++)
  	phi_1(nx,ny,nz) = nx+ny+nz + 0.4;



  std::cout << phi_1 << std::endl;

  if (phi_1.get_me() == 0)
    std::cout << "\nTesting for fftwArr of type " << typeid(T).name()
	      << " was succesful.\n" << std::endl;
  
  
}
