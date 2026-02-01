#include <mpi.h>
#include <complex>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>

#include "fftw_arr/arrays.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

double func_phi(double ,double,void * );
double func_gradphi_x(double,double,void *);
double func_gradphi_y(double,double,void *);

void initialize_gradphi(std::array<fftwArr::r2c_2D,2> &,double);

void compute_gradients(std::array<fftwArr::c2r_2D,2> &,
		       const fftwArr::c2r_2D &,
		       double); 
void save_outputs(const fftwArr::r2c_2D &,
		  const std::array<fftwArr::r2c_2D,2> &,
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
  int Ny = 20;

  double L = 2*M_PI;

  std::array<double,2> differentials = {L/Nx,L/Ny};
  std::array<double,2> origin = {0.0,0.0};
  

  // define the array to be transform, phi(x,y,z), in both real and fourier space
  fftwArr::r2c_2D phi(MPI_COMM_WORLD,"phi",Nx,Ny);


  fftwArr::c2r_2D ft_phi(MPI_COMM_WORLD,"ft_phi",Nx,Ny);
  
  // define arrays to hold the gradients in both real and fourier space
  std::array<fftwArr::r2c_2D,2> gradphi;
  std::array<fftwArr::c2r_2D,2> ft_gradphi;

  std::array<fftwArr::r2c_2D,2> errors_gradphi;

  std::array<std::string,2> xy = {"x","y"};

  for (int dim = 0; dim < 2; dim++) {
    gradphi[dim]
      = fftwArr::r2c_2D(MPI_COMM_WORLD,"gradphi_"+xy[dim],Nx,Ny);
    ft_gradphi[dim]
      = fftwArr::c2r_2D(MPI_COMM_WORLD,"ft_gradphi_"+xy[dim],Nx,Ny);
    errors_gradphi[dim]
      = fftwArr::r2c_2D(MPI_COMM_WORLD,"errors_"+xy[dim],Nx,Ny);
    
  }


  // boiler plate fftw3 stuff here
  
  fftw_plan forward_phi
    = fftw_mpi_plan_dft_r2c_2d(Ny,Nx,phi.data(),
			       reinterpret_cast<fftw_complex*>
			       (ft_phi.data()),
			       MPI_COMM_WORLD,FFTW_ESTIMATE);

  
  std::array<fftw_plan,2> backward_gradphi;


  for (int dim = 0; dim < 2; dim++)
    backward_gradphi[dim]
      = fftw_mpi_plan_dft_c2r_2d(Ny,Nx,
				 reinterpret_cast<fftw_complex*>
				 (ft_gradphi[dim].data()),
				 gradphi[dim].data(),MPI_COMM_WORLD,
				 FFTW_ESTIMATE);

  // initialise data for phi and compute fourier transform
  phi.apply_function(func_phi,nullptr,differentials,origin);
  fftw_execute(forward_phi);

  // compute gradients in fourier space
  compute_gradients(ft_gradphi,ft_phi,L);
  //compute_gradients(ft_gradphi,ft_phi,L,Nx,Ny);
  
  // inverse fourier transform to get gradients in real space

  for (int dim = 0; dim < 2; dim++) {
    fftw_execute(backward_gradphi[dim]);
  }
  for (int dim = 0; dim < 2; dim++) {
    gradphi[dim] /= Nx*Ny;
  }

  // compute analytic form of gradient, store in errors for now

  errors_gradphi[0].apply_function(func_gradphi_x,nullptr,
				   differentials,origin);

  errors_gradphi[1].apply_function(func_gradphi_y,nullptr,
				     differentials,origin);


  // subtract the computed gradients from the analytic gradients

  for (int dim = 0; dim < 2; dim++)
    errors_gradphi[dim] -= gradphi[dim];

  int global_flag;

  int local_flag;
  for (int dim = 0; dim < 2; dim++)
    local_flag
      = fftwArrTestingUtils::all_zero_2d(errors_gradphi[dim],tolerance);
  

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
    std::cout << "\n\nSUCCESSFULLY CALCULATED THE GRADIENT OF FUNCTION "
	      << "sin(x)*cos(y) USING FOURIER TRANSFORMS "
	      << "(within an absolute error of " << tolerance << ").\n\n"
	      << std::endl;


  /*  */
  if (argc > 2)
    save_data = true;
  else if (phi.get_me() == 0)
    std::cout << "\n\nTo save data for visualisation, add a second "
	      << "argument to the executable call, e.g. "
	      << std::string(argv[0]) << std::string(" 1e-4 SAVE")
	      << "\n\n" << std::endl;


  if (save_data)
    save_outputs(phi,gradphi,L);


  fftw_destroy_plan(forward_phi);

  for (int i = 0; i < 2; i++) {
    fftw_destroy_plan(backward_gradphi[i]);
  }
  
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

double func_phi(double x,double y,void *obj)
{
  return sin(x)*cos(y);
}

double func_gradphi_x(double x, double y,void *obj)
{

  return cos(x)*cos(y);

}

double func_gradphi_y(double x, double y,void *obj)
{

  return -sin(x)*sin(y);

}



void save_outputs(const fftwArr::r2c_2D &phi,
		  const std::array<fftwArr::r2c_2D,2> &gradphi,
		  double L)
{


  int me = phi.get_me();
  int nprocs = phi.get_nprocs();

  std::ofstream myfile("output_" + std::to_string(me) + ".txt");



  int Nx = phi.global_Nx();
  int Ny = phi.global_Ny();
  
  double dx = L/Nx;
  double dy = L/Ny;
  

  int local0start = phi.get_local0start();

  double x,y;


  myfile << "x,y,phi,gradphi_x,gradphi_y" << std::endl;
  
  for (int jy = 0; jy < phi.size_axis1(); jy++) {
    y = (jy+local0start)*dy;
    for (int ix = 0; ix < phi.size_axis0(); ix++) {
      x = ix*dx;
      myfile << x << "," << y << "," << phi(ix,jy) << ","
	     << gradphi[0](ix,jy) << "," << gradphi[1](ix,jy) << std::endl;
    }
  }
  
  
  
}
std::complex<double> get_qx(double qx, double qy, void *obj)
{

  return qx;
}


std::complex<double> get_qy(double qx, double qy, void *obj)
{
  
  return qy;
}




void compute_gradients(std::array<fftwArr::c2r_2D,2> &ft_gradphi,
		       const fftwArr::c2r_2D &ft_phi,  double L)
{

  // this is redundant, just highlighting the feeatures of the array2D class

  std::complex<double> I1(0,1);

  std::array<double,2> differentials = {2*M_PI/L,2*M_PI/L};

  ft_gradphi[0].apply_function(get_qx,nullptr,differentials);
  

  ft_gradphi[1].apply_function(get_qy,nullptr,differentials);


  for (int dim = 0; dim < 2; dim++) {
    ft_gradphi[dim] *= ft_phi;
    ft_gradphi[dim] *= I1;
  }


  return;
}

