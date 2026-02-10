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
void test_function(MPI_Comm ,int ,const std::string &,
		   enum fftwArr::Transposed);

int main()
{
  
  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);


  fftw_mpi_init();

  std::string file_directory;



  
  if (me == 0) {
    std::cout << "Please specify the directory to save to : ";
    std::cin >> file_directory;
  }

  int fd_string_length = file_directory.size();  
  MPI_Bcast(&fd_string_length, 1, MPI_INT, 0, world);

  if (me != 0)
    file_directory.resize(fd_string_length);

  MPI_Bcast(const_cast<char*>(file_directory.data()),
	    fd_string_length, MPI_CHAR, 0, world);
  
  for (int dim = 2; dim <=3 ; dim ++) {
  

    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,file_directory,fftwArr::Transposed::NO);

    if (dim != 2)
      test_function<
	fftwArr::Transform::C2R,std::complex<double>
	>(world,dim,file_directory,fftwArr::Transposed::YES);

    
    
  }
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}




template < enum fftwArr::Transform rOc,typename T>
void test_function(MPI_Comm world,int dim,
		   const std::string &directory,
		   enum fftwArr::Transposed transpose)
{

  std::unique_ptr<fftwArr::array2D<rOc,T>> phi_2d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> phi_3d;
  std::unique_ptr<fftwArr::ConjugateSymmetryBase<rOc,T>> conj;


  int me;
  MPI_Comm_rank(world,&me);

  std::vector<int> split_sizes;
  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());

  std::string filename, output;

  bool is_transposed;
  
  if (dim == 2) {
    
    
    int Nx = 17;
    int Ny = 20;

    
    phi_2d =
      std::make_unique<
	fftwArr::array2D<rOc,T>
	>(world,"phi_2d",Nx,Ny,transpose);
    

    conj =
      std::make_unique<
	fftwArr::ConjugateSymmetryBase<rOc,T>
	>(phi_2d->size_axis1(),phi_2d->get_local0start(),
	  phi_2d->get_world());


    
    is_transposed = phi_2d->is_transposed();

    
  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 19;
    int Nz = 10;

    
    phi_3d =
      std::make_unique<
	fftwArr::array3D<rOc,T>
	>(world,"phi_3d",Nx,Ny,Nz,transpose);

    conj =
      std::make_unique<
	fftwArr::ConjugateSymmetryBase<rOc,T>
	>(phi_3d->size_axis2(),phi_3d->get_local0start(),
	  phi_3d->get_world());

    
    is_transposed = phi_3d->is_transposed();

    
  }



  filename = directory + std::string("/") +
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
  
  
  
  
  
  writefile << conj->print_details();

  
  
  
  
  
  if (me == 0) {
    
    std::cout
      << "SUCCESS: "
      + fftwArrTestingUtils::fftwArrName(dtype,rOc,dim,
					 is_transposed)
      << std::endl;

    
  }

}

