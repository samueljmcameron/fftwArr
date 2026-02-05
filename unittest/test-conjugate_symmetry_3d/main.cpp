#include <mpi.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <complex>
#include <vector>
#include <memory>

#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"
#include "fftw_arr_testing_utils/utils.hpp"
#include "fftw_arr/conjugate_symmetry_base.hpp"

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


  

  for (int dim = 3; dim <=3 ; dim ++) {
  

    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO);

    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES);

    /*
    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO);

    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES);
    */	
    
    
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
  std::unique_ptr<fftwArr::ConjugateSymmetryBase<rOc,T>> conj_3d;


  int me;
  MPI_Comm_rank(world,&me);

  std::vector<int> split_sizes;
  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());

  std::string filename, output;

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

    conj_3d =
      std::make_unique<
	fftwArr::ConjugateSymmetryBase<rOc,T>
	>(phi_3d->size_axis2(),phi_3d->get_local0start(),
	  phi_3d->get_world());

    
    is_transposed = phi_3d->is_transposed();

    filename = "output/" + 
      fftwArrTestingUtils::fftwArrName(dtype,rOc,dim,
				       is_transposed);
    
    filename += std::string("_p") + std::to_string(me);
    std::ofstream writefile;

    writefile.open(filename);
    
    
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

    


    
    writefile << conj_3d->print_details();

    if (me == 0) {
      std::cout << "left bounds globally:" << std::endl;
      for (auto & arr : conj_3d->list_of_left_bounds)
	for (auto item : arr)
	  std::cout << item << " ";
      std::cout << std::endl;

      std::cout << "right bounds globally:" << std::endl;
      for (auto & arr : conj_3d->list_of_right_bounds)
	for (auto item : arr)
	  std::cout << item << " ";
      std::cout << std::endl;
    }
    
    /*
    if (me == 0) {
      std::cout << "SEND TO LIST: " << std::endl;
      for (auto s : conj_3d->send_to_list)
	std::cout << s << std::endl;
    }
    */
    
    
  }

  
  
  
  
  
  if (me == 0) {
    
    std::cout
      << "SUCCESS: "
      + fftwArrTestingUtils::fftwArrName(dtype,rOc,dim,
					 is_transposed)
      << std::endl;

    
  }

}

