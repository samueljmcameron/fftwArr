#include <mpi.h>
#include <iostream>
#include <tuple>
#include <complex>
#include "fftw_arr/array2d.hpp"
#include "fftw_arr/array3d.hpp"

template <typename T>
void test_function(MPI_Comm, int);

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


  for (int dim = 2; dim <= 3; dim++) {
  // std::apply invokes the lambda function with elements of type as arguments
  
  // NOTE: can't use && as that just provides a reference to the tuple elements,
  // which would make e.g. decltype(std::get<0>(types)) return &double instead of
  // double
    std::apply([=](auto... args){((test_function<decltype(args)>(world,dim)),...);},types);  

  }
  //test_function<double>(world);

  //test_function<std::complex<double>>(world);
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

template <typename T>
void test_function(MPI_Comm world,int dim)
{


  int local_flag,global_flag;
  std::string broken_file = "";
  local_flag = 0;

  if (dim == 2) {
    int Nx = 3;
    int Ny = 13;
    
    
    
    fftwArr::array2D<T> phi_1(world,"phi_1",Nx,Ny);
    
    fftwArr::array2D<T> phi_2(world,"phi_2",Nx,Ny);

    for (int ny = 0; ny < phi_1.Ny(); ny++)
      for (int nx = 0; nx < phi_1.Nx(); nx++)
	phi_1(nx,ny) = nx+ny + 0.4;
    
    
    std::fstream writefile;
    
    std::string filename = std::string("generated_files/");
    filename = filename + typeid(T).name() + std::string("dim_")
      + std::to_string(dim) + std::string("_p")
      + std::to_string(phi_1.get_me()) + std::string(".out");
    
    writefile.open(filename,
		   std::fstream::binary | std::fstream::out);



    if (!writefile) {
      local_flag = 1;
      broken_file = "Failed to open (write-only) " + filename;
    }


    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error(broken_file);

    
    phi_1.write_to_binary(writefile);
    
    
    writefile.close();
    
    std::fstream readfile;
    
    readfile.open(filename,
		  std::fstream::binary | std::fstream::in);
    

    if (!readfile) {
      local_flag = 1;
      broken_file = "Failed to open (read-only) " + filename;
    }


    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error(broken_file);
    
    phi_2.read_from_binary(readfile);
    
    for (int ny = 0; ny < phi_2.Ny(); ny ++ )
      for (int nx = 0; nx < phi_2.Nx(); nx++)
	if (std::abs(phi_2(nx,ny)-phi_1(nx,ny)) > 1e-3)
	  local_flag = 1;


    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error("MISMATCHED I/O");
    
    
    if (phi_1.get_me() == 0)
      std::cout << "\nTesting for 2D fftwArr of type " << typeid(T).name()
		<< " was succesful.\n" << std::endl;
  } else if (dim == 3) {
  
    int Nx = 3;
    int Ny = 2;
    int Nz = 10;
    
    
    
    fftwArr::array3D<T> phi_1(world,"phi_1",Nx,Ny,Nz);
    
    fftwArr::array3D<T> phi_2(world,"phi_2",Nx,Ny,Nz);
    
    for (int nz = 0; nz < phi_1.Nz(); nz++)
      for (int ny = 0; ny < phi_1.Ny(); ny++)
	for (int nx = 0; nx < phi_1.Nx(); nx++)
	  phi_1(nx,ny,nz) = nx+ny+nz + 0.4;
    
    
    std::fstream writefile;
    
    std::string filename = std::string("generated_files/");
    filename = filename + typeid(T).name() + std::string("dim_")
      + std::to_string(dim) + std::string("_p")
      + std::to_string(phi_1.get_me()) + std::string(".out");
    
    writefile.open(filename,
		   std::fstream::binary | std::fstream::out);



    if (!writefile) {
      local_flag = 1;
      broken_file = "Failed to open (write-only) " + filename;
    }


    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error(broken_file);

    
    phi_1.write_to_binary(writefile);
    
    
    writefile.close();
    
    std::fstream readfile;
    
    readfile.open(filename,
		  std::fstream::binary | std::fstream::in);
    

    if (!readfile) {
      local_flag = 1;
      broken_file = "Failed to open (read-only) " + filename;
    }


    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error(broken_file);
    
    phi_2.read_from_binary(readfile);
    
    for (int nz = 0; nz < phi_2.Nz(); nz ++ )
      for (int ny = 0; ny < phi_2.Ny(); ny ++ )
	for (int nx = 0; nx < phi_2.Nx(); nx++)
	  if (std::abs(phi_2(nx,ny,nz)-phi_1(nx,ny,nz)) > 1e-3)
	    local_flag = 1;

    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT,MPI_SUM,
		  phi_1.get_world());

    if (global_flag > 0)
      throw std::runtime_error("MISMATCHED I/O");
    
    
    if (phi_1.get_me() == 0)
      std::cout << "\nTesting for 3D fftwArr of type " << typeid(T).name()
		<< " was succesful.\n" << std::endl;
  }
  
}
