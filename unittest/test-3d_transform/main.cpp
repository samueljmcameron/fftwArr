#include <mpi.h>
#include <complex>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>

#include "fftw_arr/arrays.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

double func_phi(double ,double ,double,void * );
double func_gradphi_x(double,double,double,void *);
double func_gradphi_y(double,double,double,void *);
double func_gradphi_z(double,double,double,void *);

void initialize_gradphi(std::array<fftwArr::r2c_3D,3> &,double);

void compute_gradients(std::array<fftwArr::c2r_3D,3> &,
		       const fftwArr::c2r_3D &,
		       double); 
void save_outputs(const fftwArr::r2c_3D &,
		  const std::array<fftwArr::r2c_3D,3> &,
		  double);

int main(int argc, char **argv)
{
  
  double tolerance;

  try {
    tolerance = std::stod(argv[1]);
  } catch (const std::logic_error &e) {
    throw std::runtime_error("Need to input a tolerance for the calculation"
			     + std::string(", e.g. ")
			     + std::string(argv[0]) + std::string(" 1e-4"));
  }

  bool save_data = false;
  
  
  int ierr = MPI_Init(NULL,NULL);

  fftw_mpi_init();

  int Nx = 40;
  int Ny = 34;
  int Nz = 20;
  double L = 2*M_PI;

  std::array<double,3> differentials = {L/Nx,L/Ny,L/Nz};
  std::array<double,3> origin = {0.0,0.0,0.0};
  

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::r2c_3D phi(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);


  fftwArr::c2r_3D ft_phi(MPI_COMM_WORLD,"ft_phi",Nx,Nz,Ny);
  
  // define arrays to hold the gradients in both real and fourier space
  std::array<fftwArr::r2c_3D,3> gradphi;
  std::array<fftwArr::c2r_3D,3> ft_gradphi;

  std::array<fftwArr::r2c_3D,3> errors_gradphi;

  std::array<std::string,3> xyz = {"x","y","z"};

  for (int dim = 0; dim < 3; dim++) {
    gradphi[dim]
      = fftwArr::r2c_3D(MPI_COMM_WORLD,"gradphi_"+xyz[dim],Nx,Ny,Nz);
    ft_gradphi[dim]
      = fftwArr::c2r_3D(MPI_COMM_WORLD,"ft_gradphi_"+xyz[dim],Nx,Nz,Ny);
    errors_gradphi[dim]
      = fftwArr::r2c_3D(MPI_COMM_WORLD,"errors_"+xyz[dim],Nx,Ny,Nz);
    
  }


  // boiler plate fftw3 stuff here
  
  fftw_plan forward_phi
    = fftw_mpi_plan_dft_r2c_3d(Nz,Ny,Nx,phi.data(),
			       reinterpret_cast<fftw_complex*>
			       (ft_phi.data()),
			       MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT);

  
  std::array<fftw_plan,3> backward_gradphi;


  for (int dim = 0; dim < 3; dim++)
    backward_gradphi[dim]
      = fftw_mpi_plan_dft_c2r_3d(Nz,Ny,Nx,
				 reinterpret_cast<fftw_complex*>
				 (ft_gradphi[dim].data()),
				 gradphi[dim].data(),MPI_COMM_WORLD,
				 FFTW_MPI_TRANSPOSED_IN);

  // initialise data for phi and compute fourier transform
  phi.apply_function(func_phi,nullptr,differentials,origin);
  fftw_execute(forward_phi);

  // compute gradients in fourier space
  compute_gradients(ft_gradphi,ft_phi,L);
  // inverse fourier transform to get gradients in real space

  for (int dim = 0; dim < 3; dim++) {
    fftw_execute(backward_gradphi[dim]);
  }
  for (int dim = 0; dim < 3; dim++) {
    gradphi[dim] /= Nx*Ny*Nz;
  }

  // compute analytic form of gradient, store in errors for now

  errors_gradphi[0].apply_function(func_gradphi_x,nullptr,
				   differentials,origin);

  errors_gradphi[1].apply_function(func_gradphi_y,nullptr,
				     differentials,origin);

  errors_gradphi[2].apply_function(func_gradphi_z,nullptr,
				     differentials,origin);


  // subtract the computed gradients from the analytic gradients

  for (int dim = 0; dim < 3; dim++)
    errors_gradphi[dim] -= gradphi[dim];

  int global_flag;

  int local_flag;
  for (int dim = 0; dim < 3; dim++)
    local_flag
      = fftwArrTestingUtils::all_zero(errors_gradphi[dim],tolerance);
  

  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		phi.get_world());

  if (global_flag) {
    std::string error_message
      = "Errors in computed gradients are larger than "
      + std::to_string(tolerance) + " for "
      + std::to_string(local_flag) + " elements of the array "
      + "on processor " + std::to_string(phi.get_me()) ;
    
    throw std::runtime_error(error_message);
  }


  if (phi.get_me() == 0)
    std::cout << "SUCCESSFULLY CALCULATED THE GRADIENT OF FUNCTION "
	      << "sin(x)*cos(y)*sin(2*z) USING FOURIER TRANSFORMS "
	      << "(within an absolute error of " << tolerance << ")."
	      << std::endl;



  if (argc > 2)
    save_data = true;
  else
    std::cout << "To save data for visualisation, add a second "
	      << "argument to the executable call, e.g. "
	      << std::string(argv[0]) << std::string(" 1e-4 SAVE")
	      << std::endl;

  
  if (save_data)
    save_outputs(phi,gradphi,L);


  fftw_destroy_plan(forward_phi);

  for (int i = 0; i < 3; i++) {
    fftw_destroy_plan(backward_gradphi[i]);
  }
  
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

double func_phi(double x,double y,double z,void *obj)
{
  return sin(x)*cos(y)*sin(2*z);
}

double func_gradphi_x(double x, double y, double z, void *obj)
{

  return cos(x)*cos(y)*sin(2*z);

}

double func_gradphi_y(double x, double y, double z, void *obj)
{

  return -sin(x)*sin(y)*sin(2*z);

}


double func_gradphi_z(double x, double y, double z,void *obj)
{

  return 2*sin(x)*cos(y)*cos(2*z);	

}


void save_outputs(const fftwArr::r2c_3D &phi,
		  const std::array<fftwArr::r2c_3D,3> &gradphi,
		  double L)
{


  int me = phi.get_me();
  int nprocs = phi.get_nprocs();

  std::ofstream myfile("output_" + std::to_string(me) + ".txt");



  int Nx = phi.global_Nx();
  int Ny = phi.global_Ny();
  int Nz = phi.global_Nz();
  
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
	myfile << x << "," << y << "," << z << "," << phi(k,j,i) << ","
	       << gradphi[0](k,j,i) << "," << gradphi[1](k,j,i) << ","
	       << gradphi[2](k,j,i) << std::endl;
      }
    }

  }

}
std::complex<double> get_qx(double qx, double qy, double qz, void *obj)
{

  return qx;
}


std::complex<double> get_qy(double qx, double qy, double qz, void *obj)
{
  
  return qy;
}


std::complex<double> get_qz(double qx, double qy, double qz, void *obj)
{

  return qz;
}


void compute_gradients(std::array<fftwArr::c2r_3D,3> &ft_gradphi,
		       const fftwArr::c2r_3D &ft_phi,  double L)
{

  // this is redundant, just highlighting the feeatures of the array3D class

  std::complex<double> I1(0,1);

  std::array<double,3> differentials = {2*M_PI/L,2*M_PI/L,2*M_PI/L};

  ft_gradphi[0].apply_function(get_qx,nullptr,differentials);
  
  // swap Y and Z since dealing with transpose!
  ft_gradphi[1].apply_function(get_qz,nullptr,differentials);
  ft_gradphi[2].apply_function(get_qy,nullptr,differentials);

  for (int dim = 0; dim < 3; dim++) {
    ft_gradphi[dim] *= ft_phi;
    ft_gradphi[dim] *= I1;
  }


  return;
}



