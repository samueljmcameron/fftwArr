#include <mpi.h>
#include <iostream>
#include <complex>
#include <memory>
#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

template < enum fftwArr::Transform rOc, typename T>
void test_function(MPI_Comm ,int,
		   enum fftwArr::Transposed,
		   double );

template <typename T>
T func_2d(double x, double y, void *obj)
{
  return  x+y+ 0.4;
}

template <typename T>
T func_3d(double x, double y, double z,void *obj)
{
  return  x+y+z+ 0.4;
}



int main()
{


  int ierr = MPI_Init(NULL,NULL);

  MPI_Comm world = MPI_COMM_WORLD;
  
  int me, nprocs;
  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);


  fftw_mpi_init();



  double tolerance = 1e-5;

  for (int dim = 2; dim <= 3; dim++) {
    test_function<
      fftwArr::Transform::R2C,double
      >(world,dim,fftwArr::Transposed::NO,tolerance);
    
    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO,tolerance);
    
    test_function<
      fftwArr::Transform::C2R,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES,tolerance);
    
    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::NO,tolerance);
    
    test_function<
      fftwArr::Transform::C2C,std::complex<double>
      >(world,dim,fftwArr::Transposed::YES,tolerance);
    
    
  }

  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

template < enum fftwArr::Transform rOc,typename T>
void test_function(MPI_Comm world,int dim,
		   enum fftwArr::Transposed transpose,
		   double tolerance)
{

  int me;
  MPI_Comm_rank(world,&me);

  bool is_transposed;

  if (transpose == fftwArr::Transposed::YES)
    is_transposed = true;
  else
    is_transposed = false;

  
  std::string dtype = fftwArrTestingUtils::TypeToString(typeid(T).name());


  /* First block of the code is to open a file which the array data
     will be written to. */
    
  std::string filename = std::string("generated_files/");
  std::string arrname =
    fftwArrTestingUtils::fftwArrName(dtype,rOc,dim,is_transposed);

  
  filename =
    filename + arrname + "_p" + std::to_string(me)
    +  std::string(".out");

  std::fstream writefile;
  
  writefile.open(filename,
		 std::fstream::binary | std::fstream::out);


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



  /* Second block of the code is to write data to the appropriate array, which
     in this case is phi_2d or phi_3d .*/
  
  std::unique_ptr<fftwArr::array2D<rOc,T>> phi_2d;
  std::unique_ptr<fftwArr::array2D<rOc,T>> theta_2d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> phi_3d;
  std::unique_ptr<fftwArr::array3D<rOc,T>> theta_3d;  


  
  if (dim == 2) {
    int Nx = 9;
    int Ny = 13;
    

    
    phi_2d =
      std::make_unique<fftwArr::array2D<rOc,T>>(world,"phi_2d",
						Nx,Ny,transpose);
    theta_2d =
      std::make_unique<fftwArr::array2D<rOc,T>>(world,"theta_2d",
						Nx,Ny,transpose);

    is_transposed = phi_2d->is_transposed();

    phi_2d->apply_function(func_2d<T>,nullptr,{1.0,1.0});
    
    
    phi_2d->write_to_binary(writefile);

    

  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 9;
    int Nz = 10;
    
    
    
    phi_3d =
      std::make_unique<fftwArr::array3D<rOc,T>>(world,"phi_3d",
						Nx,Ny,Nz,transpose);
    theta_3d =
      std::make_unique<fftwArr::array3D<rOc,T>>(world,"theta_3d",
						Nx,Ny,Nz,transpose);
    

    is_transposed = phi_3d->is_transposed();
    phi_3d->apply_function(func_3d<T>,nullptr,{1.0,1.0,1.0});
    
    phi_3d->write_to_binary(writefile);
  }

  writefile.close();



  /* Third block of the code is to open a read only file .*/

    
  std::fstream readfile;
  
  readfile.open(filename,
		std::fstream::binary | std::fstream::in);
  
  
  if (!readfile) {
    local_flag = 1;
    broken_file = "Failed to open (read-only) " + filename;
  }
  
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		world);
  
  if (global_flag > 0)
    throw std::runtime_error(broken_file);


  /* Fourth block of the code is to read data to the appropriate array, which
     in this case is theta_2d or theta_3d .*/

  
  if (dim == 2) {
    
    theta_2d->read_from_binary(readfile);

    (*theta_2d) -= (*phi_2d);
    
    local_flag = fftwArrTestingUtils::all_zero_2d(*theta_2d,
						  tolerance);
  } else if (dim == 3) {

    theta_3d->read_from_binary(readfile);

    (*theta_3d) -= (*phi_3d);

    local_flag = fftwArrTestingUtils::all_zero_3d(*theta_3d,
						  tolerance);

    
  }
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		world);
  
  if (global_flag > 0)
    throw std::runtime_error("MISMATCHED I/O");
  
  
  if (me == 0)
    std::cout
      << fftwArrTestingUtils::SuccessMessage(dtype,rOc,dim,
					     is_transposed)
      << std::endl;
  
}
