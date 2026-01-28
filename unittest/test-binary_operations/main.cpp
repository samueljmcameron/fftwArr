#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
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

template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm world,int dim)
{
  int me;
  MPI_Comm_rank(world,&me);
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());

  if (dim == 2) {
    int Nx = 5;
    int Ny = 10;
    
    // define the array to be transform, phi(x,y,z), in both real and fourier space
    fftwArr::array2D<rOc,T> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny);
    fftwArr::array2D<rOc,T> phi_2;
  
    phi_2 = fftwArr::array2D<rOc,T>(MPI_COMM_WORLD,"phi_2",Nx,Ny);

    phi_1 += 2.0;
    phi_1 /= 2.0;
    phi_1 *= 3.0;
    phi_1 -= 2.0;
    
    phi_2 += 3.0;
    
    phi_1 += phi_2;
    
  } else if (dim == 3) {
    int Nx = 5;
    int Ny = 2;
    int Nz = 20;
    
    // define the array to be transform, phi(x,y,z), in both real and fourier space
    fftwArr::array3D<rOc,T> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny,Nz);
    fftwArr::array3D<rOc,T> phi_2;
    
    phi_2 = fftwArr::array3D<rOc,T>(MPI_COMM_WORLD,"phi_2",Nx,Ny,Nz);
    
    phi_1 += 2.0;
    phi_1 /= 2.0;
    phi_1 *= 3.0;
    phi_1 -= 2.0;
    
    phi_2 += 3.0;
    
    phi_1 += phi_2;
    
  }

  if (me == 0)
    std::cout << fftwArrTestingUtils::SuccessMessage(dtype,rOc,dim) << std::endl;

  
}
