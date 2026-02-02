#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"
#include "fftw_arr_testing_utils/utils.hpp"


template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm ,int , enum fftwArr::Transposed);

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
    test_function<
      fftwArr::Transform::R2C,double
      >(world,dim,fftwArr::Transposed::NO);

    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO);

    
    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES);

    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO);

    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES);

  }
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm world,int dim,
		   enum fftwArr::Transposed transpose)
{
  int me;
  MPI_Comm_rank(world,&me);
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());

  bool is_transposed;

  if (dim == 2) {
    int Nx = 13;
    int Ny = 17;
    
    // define the array to be transform, phi(x,y,z), in both real and fourier space
    fftwArr::array2D<rOc,T> phi_1(MPI_COMM_WORLD,"phi_1",Nx,Ny,
				  transpose);
    fftwArr::array2D<rOc,T> phi_2;
  
    phi_2 = fftwArr::array2D<rOc,T>(MPI_COMM_WORLD,"phi_2",Nx,Ny,
				    transpose);

    phi_1 += 2.0;
    phi_1 /= 2.0;
    phi_1 *= 3.0;
    phi_1 -= 2.0;
    
    phi_2 += 3.0;
    
    phi_1 += phi_2;

    is_transposed = phi_2.is_transposed();
    
  } else if (dim == 3) {
    int Nx = 5;
    int Ny = 15;
    int Nz = 20;
    
    // define the array to be transform, phi(x,y,z), in both real and fourier space
    fftwArr::array3D<rOc,T> phi_1(MPI_COMM_WORLD,"phi_1",
				  Nx,Ny,Nz,transpose);
    fftwArr::array3D<rOc,T> phi_2;
    
    phi_2 = fftwArr::array3D<rOc,T>(MPI_COMM_WORLD,"phi_2",
				    Nx,Ny,Nz,transpose);
    
    phi_1 += 2.0;
    phi_1 /= 2.0;
    phi_1 *= 3.0;
    phi_1 -= 2.0;
    
    phi_2 += 3.0;
    
    phi_1 += phi_2;
    is_transposed = phi_2.is_transposed();    
  }

  if (me == 0)
    std::cout
      << fftwArrTestingUtils::SuccessMessage(dtype,rOc,dim,
					     is_transposed)
      << std::endl;

  
}
