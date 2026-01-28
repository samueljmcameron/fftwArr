#include <mpi.h>
#include <iostream>
#include <tuple>
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

  // what follows is a hack to iterate over different types.

  // first, define a tuple with all the types you want to iterate over
  // std::tuple<double,std::complex<double>> types;
  // std::apply invokes the lambda function with elements of type as arguments
  
  // NOTE: can't use && as that just provides a reference to the tuple elements,
  // which would make e.g. decltype(std::get<0>(types)) return &double instead of
  // double
  //std::apply([=](auto... args){((test_function<decltype(args)>(world,dim)),...);},types);

  for (int dim = 2; dim <=3 ; dim ++) {
  
    test_function<fftwArr::Transform::R2C,double>(world,dim);

    test_function<fftwArr::Transform::C2R,std::complex<double>>(world,dim);

    test_function<fftwArr::Transform::C2C,std::complex<double>>(world,dim);

    
    
  }
  
  fftw_mpi_cleanup();


  
  if (nprocs > 1)
    std::cout << "WARNING: ostream output will be out of order in general for nprocs > 1." << std::endl;


  ierr = MPI_Finalize();

  return 0;
}




template < enum fftwArr::Transform rOc,typename T>
void test_function(MPI_Comm world,int dim)
{

  std::unique_ptr<fftwArr::array2D<rOc,T>> phi_2d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> phi_3d;

  phi_2d = nullptr;
  phi_3d = nullptr;

  int me;
  MPI_Comm_rank(world,&me);
  
  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());
  
  if (dim == 2) {
    
    
    int Nx = 3;
    int Ny = 9;
    
    
    phi_2d = std::make_unique<fftwArr::array2D<rOc,T>>(world,"phi_2d",Nx,Ny);
    
    for (int ny = 0; ny < phi_2d->Ny(); ny++)
      for (int nx = 0; nx < phi_2d->Nx(); nx++)
	(*phi_2d)(nx,ny) = nx+ny + 0.4;
    
    std::cout << *phi_2d << std::endl;
    
  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 2;
    int Nz = 10;
    
    phi_3d = std::make_unique<fftwArr::array3D<rOc,T>>(world,"phi_3d",Nx,Ny,Nz);

    
    for (int nz = 0; nz < phi_3d->Nz(); nz++)
      for (int ny = 0; ny < phi_3d->Ny(); ny++)
	for (int nx = 0; nx < phi_3d->Nx(); nx++)
	  (*phi_3d)(nx,ny,nz) = nx+ny+nz + 0.4;
    
    
    
    std::cout << *phi_3d << std::endl;

  }
  if (me == 0)
    std::cout << "SUCCESS: " + fftwArrTestingUtils::fftwArrName(dtype,rOc,dim) << std::endl;

}

