#include <mpi.h>
#include <complex>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>

#include "fftw_arr/r2c_2d.hpp"


void initialize_phi(fftwArr::r2c2d<double> &,double ,int,int);
void initialize_gradphi(std::array<fftwArr::r2c2d<double>,3> &,double,
			int , int);

void compute_gradients(std::array<fftwArr::r2c2d<std::complex<double>>,2> &,
		       const fftwArr::r2c2d<std::complex<double>> &,
		       double,int,int); 
void save_outputs(const fftwArr::r2c2d<double> &,
		  const std::array<fftwArr::r2c2d<double>,2> &,
		  double, int, int);


int main()
{
  

  
  int ierr = MPI_Init(NULL,NULL);

  fftw_mpi_init();

  int Nx = 36;
  int Ny = 20;
  double L = 2*M_PI;

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::r2c2d<double> phi;
  phi.assign(MPI_COMM_WORLD,"phi",Nx,Ny);

  
  fftwArr::r2c2d<std::complex<double>> ft_phi;
  ft_phi.assign(MPI_COMM_WORLD,"ft_phi",Ny,Nx);

  // define arrays to hold the gradients in both real and fourier space
  std::array<fftwArr::r2c2d<double>,2> gradphi;
  std::array<fftwArr::r2c2d<std::complex<double>>,2> ft_gradphi;


  std::array<std::string,3> xy = {"x","y"};

  for (int i = 0; i < 2; i++) {
    gradphi[i].assign(MPI_COMM_WORLD,"gradphi_"+xy[i],Nx,Ny);
    ft_gradphi[i].assign(MPI_COMM_WORLD,"ft_gradphi_"+xy[i],Ny,Nx);
  }

  // boiler plate fftw3 stuff here
  
  fftw_plan forward_phi = fftw_mpi_plan_dft_r2c_2d(Ny,Nx,phi.data(),
						   reinterpret_cast<fftw_complex*>
						   (ft_phi.data()),
						   MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT);

  
  std::array<fftw_plan,3> backward_gradphi;


  for (int i = 0; i < 3; i++)
    backward_gradphi[i] = fftw_mpi_plan_dft_c2r_2d(Ny,Nx,
						   reinterpret_cast<fftw_complex*>
						   (ft_gradphi[i].data()),
						   gradphi[i].data(),MPI_COMM_WORLD,
						   FFTW_MPI_TRANSPOSED_IN);

  // initialise data for phi and compute fourier transform
  initialize_phi(phi,L,Nx,Ny);
  //initialize_gradphi(gradphi,L,Nx,Ny);

  fftw_execute(forward_phi);

  // compute gradients in fourier space
  compute_gradients(ft_gradphi,ft_phi,L,Nx, Ny);
  // inverse fourier transform to get gradients in real space

  for (int i = 0; i < 2; i++) {
    fftw_execute(backward_gradphi[i]);
  }
  for (int i = 0; i < 2; i++) {
    gradphi[i] /= Nx*Ny;
  }

  save_outputs(phi,gradphi,L, Nx, Ny);

  /*  */  
  fftw_destroy_plan(forward_phi);

  for (int i = 0; i < 3; i++) {
    fftw_destroy_plan(backward_gradphi[i]);
  }

  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  system("leaks sinewave-bin");

  return 0;
}

void initialize_phi(fftwArr::r2c2d<double> &phi,double L,
		    int Nx, int Ny)
{


  double dx = L/Nx;
  double dy = L/Ny;

  int local0start = phi.get_local0start();

  double x,y;

  for (int i = 0; i < phi.get_sizeax(0); i++) {
    y = (i+local0start)*dy;
    for (int j = 0; j < phi.get_sizeax(1); j++) {
      x = j*dx;
      phi(i,j) = sin(x)*cos(y);
    }

  }
}

void initialize_gradphi(std::array<fftwArr::r2c2d<double>,2> &gradphi,double L,
			int Nx, int Ny)
{
  double dx = L/Nx;
  double dy = L/Ny;

  
  int local0start = gradphi[0].get_local0start();

  double x,y;

  for (int i = 0; i < gradphi[0].get_sizeax(0); i++) {
    y = (i+local0start)*dy;
    for (int j = 0; j < gradphi[0].get_sizeax(1); j++) {
      x = j*dx;
      gradphi[0](i,j) = cos(x)*cos(y);
      gradphi[1](i,j) = -sin(x)*sin(y);

    }
  }
}


void save_outputs(const fftwArr::r2c2d<double> &phi,
		  const std::array<fftwArr::r2c2d<double>,2> &gradphi,
		  double L, int Nx, int Ny)
{


  int me,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  std::ofstream myfile("output_" + std::to_string(me) + ".txt");

  double dx = L/Nx;
  double dy = L/Ny;

  for (int proc = 0; proc < nprocs; proc++) {
    if (me == proc) {
      std::cout << "On processor " << proc << std::endl;
      std::cout << phi.get_sizeax(0) << std::endl;
      std::cout << phi.get_sizeax(1) << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int local0start = phi.get_local0start();

  double x,y;


  myfile << "x,y,phi,gradphi_x,gradphi_y" << std::endl;
  
  for (int i = 0; i < phi.get_sizeax(0); i++) {
    y = (i+local0start)*dy;
    for (int j = 0; j < phi.get_sizeax(1); j++) {
      x = j*dx;
      myfile << x << "," << y << "," << phi(i,j) << ","
	     << gradphi[0](i,j) << "," << gradphi[1](i,j) << std::endl;
    }
  }

}
		  

void compute_gradients(std::array<fftwArr::r2c2d<std::complex<double>>,2> &ft_gradphi,
		       const fftwArr::r2c2d<std::complex<double>> &ft_phi,  double L,
		       int Nx, int Ny)
{

  // this is redundant, just highlighting the feeatures of the r2c2d class

  std::complex<double> idq(0,2*M_PI/L);

  double l,m,n;

  int local0start = ft_phi.get_local0start();

  for (int i = 0; i < ft_phi.get_sizeax(0); i++) {
    if (i + local0start > Nx/2)
      l = -Nx + i + local0start;
    else
      l = i + local0start;
    for (int j = 0; j < ft_phi.get_sizeax(1); j++) {
      m = j;
      // SWAP X and Y here (m and l) since computing FFTW transpose.
	
      ft_gradphi[0](i,j) = ft_phi(i,j)*idq*l;
      ft_gradphi[1](i,j) = ft_phi(i,j)*idq*m;

    }
  }

  return;
}



