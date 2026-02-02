#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
#include <vector>
#include <memory>

#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm ,int ,
		   enum fftwArr::Transposed);

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




template < enum fftwArr::Transform rOc,typename T>
void test_function(MPI_Comm world,int dim,
		   enum fftwArr::Transposed transpose)
{

  std::unique_ptr<fftwArr::array2D<rOc,T>> phi_2d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> phi_3d;


  int me;
  MPI_Comm_rank(world,&me);

  std::vector<int> split_sizes;
  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());

  bool is_transposed;
  
  if (dim == 2) {
    
    
    int Nx = 13;
    int Ny = 9;

    
    phi_2d =
      std::make_unique<
	fftwArr::array2D<rOc,T>
	>(world,"phi_2d",Nx,Ny,transpose);
    
    split_sizes = phi_2d->split_sizes();

    is_transposed = phi_2d->is_transposed();

    
  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 19;
    int Nz = 10;

    
    phi_3d =
      std::make_unique<
	fftwArr::array3D<rOc,T>
	>(world,"phi_3d",Nx,Ny,Nz,transpose);
    
    split_sizes = phi_3d->split_sizes();
    is_transposed = phi_3d->is_transposed();
  }
  if (me == 0) {


    int global_final_axis_size = 0;
    for (auto item : split_sizes)
      global_final_axis_size += item;
    
    std::cout
      << "SUCCESS: "
      + fftwArrTestingUtils::fftwArrName(dtype,rOc,dim,
					 is_transposed)
      << std::endl;

    std::string message = "Processor split sizes along the ";
    if (dim == 2)
      message += "y axis ";
    else if (dim == 3)
      message += "z axis ";

    message += "(total length " + std::to_string(global_final_axis_size) + "): ";
    
    for (auto & item : split_sizes)
      message += std::to_string(item) + " ";

    std::cout << message << std::endl;
    
  }

}

