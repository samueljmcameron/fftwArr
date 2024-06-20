#include <mpi.h>
#include <complex>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>

#include "fftw_arr/array3d.hpp"

void initialize_phi(fftwArr::array3D<double> &,double ,int,int,int);
void initialize_gradphi(std::array<fftwArr::array3D<double>,3> &,double,
			int , int , int );

void compute_gradients(std::array<fftwArr::array3D<std::complex<double>>,3> &,
		       const fftwArr::array3D<std::complex<double>> &,
		       double,int,int,int); 
void save_outputs(const fftwArr::array3D<double> &,
		  const std::array<fftwArr::array3D<double>,3> &,
		  double, int, int, int);

int main()
{
  

  
  int ierr = MPI_Init(NULL,NULL);

  fftw_mpi_init();

  int Nx = 40;
  int Ny = 34;
  int Nz = 20;
  double L = 2*M_PI;

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::array3D<double> phi(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);

  
  fftwArr::array3D<std::complex<double>> ft_phi(MPI_COMM_WORLD,"ft_phi",Nx,Nz,Ny);

  // define arrays to hold the gradients in both real and fourier space
  std::array<fftwArr::array3D<double>,3> gradphi;
  std::array<fftwArr::array3D<std::complex<double>>,3> ft_gradphi;

  std::array<std::string,3> xyz = {"x","y","z"};

  for (int i = 0; i < 3; i++) {
    gradphi[i] = fftwArr::array3D<double>(MPI_COMM_WORLD,"gradphi_"+xyz[i],Nx,Ny,Nz);
    ft_gradphi[i] = fftwArr::array3D<std::complex<double>>
      (MPI_COMM_WORLD,"ft_gradphi_"+xyz[i],Nx,Nz,Ny);
  }


  // boiler plate fftw3 stuff here
  
  fftw_plan forward_phi = fftw_mpi_plan_dft_r2c_3d(Nz,Ny,Nx,phi.data(),
						   reinterpret_cast<fftw_complex*>
						   (ft_phi.data()),
						   MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT);

  
  std::array<fftw_plan,3> backward_gradphi;


  for (int i = 0; i < 3; i++)
    backward_gradphi[i] = fftw_mpi_plan_dft_c2r_3d(Nz,Ny,Nx,
						   reinterpret_cast<fftw_complex*>
						   (ft_gradphi[i].data()),
						   gradphi[i].data(),MPI_COMM_WORLD,
						   FFTW_MPI_TRANSPOSED_IN);


  // initialise data for phi and compute fourier transform
  initialize_phi(phi,L,Nx,Ny,Nz);
  //initialize_gradphi(gradphi,L,Nx,Ny,Nz);

  fftw_execute(forward_phi);

  // compute gradients in fourier space
  compute_gradients(ft_gradphi,ft_phi,L,Nx, Ny, Nz);
  // inverse fourier transform to get gradients in real space

  for (int i = 0; i < 3; i++) {
    fftw_execute(backward_gradphi[i]);
  }
  for (int i = 0; i < 3; i++) {
    gradphi[i] /= Nx*Ny*Nz;
  }
  


  save_outputs(phi,gradphi,L, Nx, Ny, Nz);


  fftw_destroy_plan(forward_phi);

  for (int i = 0; i < 3; i++) {
    fftw_destroy_plan(backward_gradphi[i]);
  }
  
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

void initialize_phi(fftwArr::array3D<double> &phi,double L,
		    int Nx, int Ny, int Nz)
{




  double dx = L/Nx;
  double dy = L/Ny;
  double dz = L/Nz;

  
  int local0start = phi.get_local0start();

  double x,y,z;

  for (int i = 0; i < phi.Nz(); i++) {
    z = (i+local0start)*dz;
    for (int j = 0; j < phi.Ny(); j++) {
      y = j*dy;
      for (int k = 0; k < phi.Nx(); k++) {
	x = k*dx;
	phi(i,j,k) = sin(x)*cos(y)*sin(2*z);
      }
    }

  }
}

void initialize_gradphi(std::array<fftwArr::array3D<double>,3> &gradphi,double L,
			int Nx, int Ny, int Nz)
{




  double dx = L/Nx;
  double dy = L/Ny;
  double dz = L/Nz;

  
  int local0start = gradphi[0].get_local0start();

  double x,y,z;

  for (int i = 0; i < gradphi[0].Nz(); i++) {
    z = (i+local0start)*dz;
    for (int j = 0; j < gradphi[0].Ny(); j++) {
      y = j*dy;
      for (int k = 0; k < gradphi[0].Nx(); k++) {
	x = k*dx;
	gradphi[0](i,j,k) = cos(x)*cos(y)*sin(2*z);
	gradphi[1](i,j,k) = -sin(x)*sin(y)*sin(2*z);
	gradphi[2](i,j,k) = 2*sin(x)*cos(y)*cos(2*z);	

      }
    }
  }
}


void save_outputs(const fftwArr::array3D<double> &phi,
		  const std::array<fftwArr::array3D<double>,3> &gradphi,
		  double L, int Nx, int Ny, int Nz)
{


  int me,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  std::ofstream myfile("output_" + std::to_string(me) + ".txt");

  double dx = L/Nx;
  double dy = L/Ny;
  double dz = L/Nz;
  

  int local0start = phi.get_local0start();

  double x,y,z;


  myfile << "x,y,z,phi,gradphi_x,gradphi_y,gradphi_z" << std::endl;
  
  for (int i = 0; i < phi.Nz(); i++) {
    z = (i+local0start)*dz;
    for (int j = 0; j < phi.Ny(); j++) {
      y = j*dy;
      for (int k = 0; k < phi.Nx(); k++) {
	x = k*dx;
	myfile << x << "," << y << "," << z << "," << phi(i,j,k) << ","
	       << gradphi[0](i,j,k) << "," << gradphi[1](i,j,k) << ","
	       << gradphi[2](i,j,k) << std::endl;
      }
    }

  }

  

}
		  

void compute_gradients(std::array<fftwArr::array3D<std::complex<double>>,3> &ft_gradphi,
		       const fftwArr::array3D<std::complex<double>> &ft_phi,  double L,
		       int Nx, int Ny, int Nz)
{

  // this is redundant, just highlighting the feeatures of the array3D class


  std::complex<double> idq(0,2*M_PI/L);

  double l,m,n;

  int local0start = ft_phi.get_local0start();

  for (int i = 0; i < ft_phi.Nz(); i++) {
    if (i + local0start > Ny/2)
      l = -Ny + i + local0start;
    else
      l = i + local0start;
    for (int j = 0; j < ft_phi.Ny(); j++) {
      if (j > Nz/2)
	m = -Nz + j;
      else
	m = j;
      for (int k = 0; k < ft_phi.Nx(); k++) {
	n = k;
	
	ft_gradphi[0](i,j,k) = ft_phi(i,j,k)*idq*n;

	// SWAP Y and Z here (m and l) since computing FFTW transpose.
	ft_gradphi[1](i,j,k) = ft_phi(i,j,k)*idq*l;
	ft_gradphi[2](i,j,k) = ft_phi(i,j,k)*idq*m;

      }
    }
  }



  return;
}



