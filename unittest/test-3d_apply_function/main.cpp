#include <mpi.h>
#include <complex>
#include <cmath>
#include <array>
#include <fstream>
#include <chrono>
#include <iostream>

#include "fftw_arr/arrays.hpp"
#include "fftw_arr_testing_utils/utils.hpp"

double func_phi(double ,double ,double, void* );

std::complex<double> func_ft_theta(double ,double ,double, void*);

std::complex<double> func_complex_psi(double ,double ,double,void* );



void r2c_3d_apply_function(double (*)(double,double,double,void*),
			   void*,fftwArr::r2c_3D &,
			   const std::array<double,3> &,
			   const std::array<double,3> &);

void c2r_3d_apply_function(std::complex<double>
			   (*)(double,double,double,void*),
			   void*,fftwArr::c2r_3D &,
			   const std::array<double,3> &,
			   const std::array<double,3> &);


void c2c_3d_apply_function(std::complex<double>
			   (*)(double,double,double,void*),
			   void*,fftwArr::c2c_3D &,
			   const std::array<double,3> &,
			   const std::array<double,3> &);


std::string error_message(const std::string & ,
			  double, int, int );


int main(int argc, char **argv)
{


  
  int ierr = MPI_Init(NULL,NULL);

  fftw_mpi_init();

  int local_flag,global_flag;

  double global_r2cMember_time;
  double global_r2cVanilla_time;

  double global_c2rMember_time;
  double global_c2rVanilla_time;

  double global_c2cMember_time;
  double global_c2cVanilla_time;

  

  
  int numcalls; // call the function this many times
  double tolerance = 1e-13;

  try {
    numcalls = std::stoi(argv[1]);
  } catch (const std::logic_error &e) {

    throw std::runtime_error("Need to input how many calls for the calculation"
			     + std::string(", e.g. ")
			     + std::string(argv[0]) + std::string(" 20"));
  }

  

  
  int Nx = 40;
  int Ny = 34;
  int Nz = 20;
  double L = 2*M_PI;

  double aNumber = 1;

  std::array<double,3> differentials = {L/Nx,L/Ny,L/Nz};
  std::array<double,3> origin = {0.0,0.0,0.0};
  

  fftwArr::r2c_3D phi(MPI_COMM_WORLD,"phi",Nx,Ny,Nz);
  fftwArr::r2c_3D output(MPI_COMM_WORLD,"output",Nx,Ny,Nz);

  
  fftwArr::c2r_3D ft_theta(MPI_COMM_WORLD,"ft_theta",Nx,Nz,Ny);
  fftwArr::c2r_3D ft_output(MPI_COMM_WORLD,"ft_output",Nx,Nz,Ny);

  fftwArr::c2c_3D complex_psi(MPI_COMM_WORLD,"complex_psi",Nx,Ny,Nz);
  fftwArr::c2c_3D complex_output(MPI_COMM_WORLD,"complex_output",Nx,Ny,Nz);  
  
  output.apply_function(func_phi,&aNumber,differentials,origin);

  ft_output.apply_function(func_ft_theta,&aNumber,differentials,origin);

  complex_output.apply_function(func_complex_psi,&aNumber,differentials,origin);
  
  phi = 0.0;
  ft_theta = 0.0;
  complex_psi = 0.0;
  
  std::chrono::steady_clock::time_point const time_start_r2cMember{
    std::chrono::steady_clock::now()};
  
  for (int call = 0; call < numcalls; call++)
    phi.apply_function(func_phi,&aNumber,differentials,origin);

  std::chrono::steady_clock::time_point const time_end_r2cMember{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_r2cMember{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_r2cMember -
							 time_start_r2cMember)
    .count()};
  double const latency_r2cMember{time_elapsed_r2cMember /
			      static_cast<double>(numcalls)};


  MPI_Allreduce(&latency_r2cMember, &global_r2cMember_time, 1,
		MPI_DOUBLE, MPI_SUM,
		phi.get_world());

  global_r2cMember_time /= phi.get_nprocs();
  
  phi -= output;
  local_flag = fftwArrTestingUtils::all_zero(phi,tolerance);


  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		phi.get_world());


  if (global_flag) {
    std::string message
      = error_message("r2cMember",tolerance,local_flag,
		      phi.get_me());
    throw std::runtime_error(message);
  }

  
  phi = 0.0;


  std::chrono::steady_clock::time_point const time_start_r2cVanilla{
    std::chrono::steady_clock::now()};
  
  
  for (int call = 0; call < numcalls; call++)
    r2c_3d_apply_function(func_phi,&aNumber,phi,differentials,origin);


  std::chrono::steady_clock::time_point const time_end_r2cVanilla{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_r2cVanilla{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_r2cVanilla -
							 time_start_r2cVanilla)
    .count()};
  double const latency_r2cVanilla{time_elapsed_r2cVanilla /
			      static_cast<double>(numcalls)};



  MPI_Allreduce(&latency_r2cVanilla, &global_r2cVanilla_time, 1,
		MPI_DOUBLE, MPI_SUM,
		phi.get_world());

  global_r2cVanilla_time /= phi.get_nprocs();
  
  phi -= output;
  local_flag = fftwArrTestingUtils::all_zero(phi,tolerance);
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		phi.get_world());


  if (global_flag) {
    std::string message
      = error_message("r2cVanilla",tolerance,local_flag,
		      phi.get_me());
    throw std::runtime_error(message);
  }

  if (phi.get_me() == 0) {
    std::cout << "Average latency when using the r2cMember method is "
	      << global_r2cMember_time/1000 << " us" << std::endl;

    std::cout << "Average latency when using the r2cVanilla method is "
	      << global_r2cVanilla_time/1000 << " us" << std::endl;
  }


  /* Now test c2r */



  std::chrono::steady_clock::time_point const time_start_c2rMember{
    std::chrono::steady_clock::now()};
  
  for (int call = 0; call < numcalls; call++)
    ft_theta.apply_function(func_ft_theta,&aNumber,differentials,origin);

  std::chrono::steady_clock::time_point const time_end_c2rMember{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_c2rMember{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_c2rMember -
							 time_start_c2rMember)
    .count()};
  double const latency_c2rMember{time_elapsed_c2rMember /
			      static_cast<double>(numcalls)};


  MPI_Allreduce(&latency_c2rMember, &global_c2rMember_time, 1,
		MPI_DOUBLE, MPI_SUM,
		ft_theta.get_world());

  global_c2rMember_time /= ft_theta.get_nprocs();
  
  ft_theta -= ft_output;
  local_flag = fftwArrTestingUtils::all_zero(ft_theta,tolerance);


  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		ft_theta.get_world());


  if (global_flag) {
    std::string message
      = error_message("c2rMember",tolerance,local_flag,
		      ft_theta.get_me());
    throw std::runtime_error(message);
  }

  
  ft_theta = 0.0;


  std::chrono::steady_clock::time_point const time_start_c2rVanilla{
    std::chrono::steady_clock::now()};
  
  
  for (int call = 0; call < numcalls; call++)
    c2r_3d_apply_function(func_ft_theta,&aNumber,
			  ft_theta,differentials,origin);


  std::chrono::steady_clock::time_point const time_end_c2rVanilla{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_c2rVanilla{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_c2rVanilla -
							 time_start_c2rVanilla)
    .count()};
  double const latency_c2rVanilla{time_elapsed_c2rVanilla /
			      static_cast<double>(numcalls)};



  MPI_Allreduce(&latency_c2rVanilla, &global_c2rVanilla_time, 1,
		MPI_DOUBLE, MPI_SUM,
		ft_theta.get_world());

  global_c2rVanilla_time /= ft_theta.get_nprocs();
  
  ft_theta -= ft_output;
  local_flag = fftwArrTestingUtils::all_zero(ft_theta,tolerance);
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		ft_theta.get_world());


  if (global_flag) {
    std::string message
      = error_message("c2rVanilla",tolerance,local_flag,
		      ft_theta.get_me());
    throw std::runtime_error(message);
  }

  if (ft_theta.get_me() == 0) {
    std::cout << "Average latency when using the c2rMember method is "
	      << global_c2rMember_time/1000 << " us" << std::endl;

    std::cout << "Average latency when using the c2rVanilla method is "
	      << global_c2rVanilla_time/1000 << " us" << std::endl;
  }

  /* now test c2c */



  std::chrono::steady_clock::time_point const time_start_c2cMember{
    std::chrono::steady_clock::now()};
  
  for (int call = 0; call < numcalls; call++)
    complex_psi.apply_function(func_complex_psi,&aNumber,differentials,origin);

  std::chrono::steady_clock::time_point const time_end_c2cMember{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_c2cMember{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_c2cMember -
							 time_start_c2cMember)
    .count()};
  double const latency_c2cMember{time_elapsed_c2cMember /
			      static_cast<double>(numcalls)};


  MPI_Allreduce(&latency_c2cMember, &global_c2cMember_time, 1,
		MPI_DOUBLE, MPI_SUM,
		complex_psi.get_world());

  global_c2cMember_time /= complex_psi.get_nprocs();
  
  complex_psi -= complex_output;
  local_flag = fftwArrTestingUtils::all_zero(complex_psi,tolerance);


  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		complex_psi.get_world());


  if (global_flag) {
    std::string message
      = error_message("c2cMember",tolerance,local_flag,
		      complex_psi.get_me());
    throw std::runtime_error(message);
  }

  
  complex_psi = 0.0;


  std::chrono::steady_clock::time_point const time_start_c2cVanilla{
    std::chrono::steady_clock::now()};
  
  
  for (int call = 0; call < numcalls; call++)
    c2c_3d_apply_function(func_complex_psi,&aNumber,complex_psi,differentials,origin);


  std::chrono::steady_clock::time_point const time_end_c2cVanilla{
    std::chrono::steady_clock::now()};
  auto const time_elapsed_c2cVanilla{
    std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_c2cVanilla -
							 time_start_c2cVanilla)
    .count()};
  double const latency_c2cVanilla{time_elapsed_c2cVanilla /
			      static_cast<double>(numcalls)};



  MPI_Allreduce(&latency_c2cVanilla, &global_c2cVanilla_time, 1,
		MPI_DOUBLE, MPI_SUM,
		complex_psi.get_world());

  global_c2cVanilla_time /= complex_psi.get_nprocs();
  
  complex_psi -= complex_output;
  local_flag = fftwArrTestingUtils::all_zero(complex_psi,tolerance);
  
  MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM,
		complex_psi.get_world());


  if (global_flag) {
    std::string message
      = error_message("c2cVanilla",tolerance,local_flag,
		      complex_psi.get_me());
    throw std::runtime_error(message);
  }

  if (complex_psi.get_me() == 0) {
    std::cout << "Average latency when using the c2cMember method is "
	      << global_c2cMember_time/1000 << " us" << std::endl;

    std::cout << "Average latency when using the c2cVanilla method is "
	      << global_c2cVanilla_time/1000 << " us" << std::endl;
  }
  
  
  fftw_mpi_cleanup();

  ierr = MPI_Finalize();

  return 0;
}

std::string error_message(const std::string & calctype,
			  double tolerance, int local_flag, int me)
{
  std::string out
    = "The original array and the called back array calculated via "
    + calctype + " don't match within tolerance "
    + std::to_string(tolerance) + " for "
    + std::to_string(local_flag) + " elements of the array "
    + "on processor " + std::to_string(me);

  return out;
}
 
void r2c_3d_apply_function(double (*func)(double,double,double,void*),
			   void *object,fftwArr::r2c_3D & output,
			   const std::array<double,3> & differential,
			   const std::array<double,3> & origin)
{

  int local_0_start = output.get_local0start();
  double x, y, z;
  
  for (int kz = 0; kz < output.Nz(); kz ++ ) {
    z = ( kz + local_0_start ) *differential[2] + origin[2];
    for (int jy = 0; jy < output.Ny(); jy ++ ) {
      y =  jy * differential[1] + origin[1];
      for (int ix = 0; ix < output.Nx(); ix ++ ) {
	x = ix * differential[0] + origin[0];
	
	output(ix,jy,kz) = func(x,y,z,object);
	
      }
    }
  }
  
  
  return;

}


void c2r_3d_apply_function(std::complex<double>
			   (*func)(double,double,double, void *),
			   void *object,fftwArr::c2r_3D & output,
			   const std::array<double,3> & differential,
			   const std::array<double,3> & origin)
{
  double qx,qy,qz;
  int global_z_size = output.global_Nz();
  int global_y_size = output.global_Ny();
  int global_x_size = output.global_Nx();
  int local_0_start = output.get_local0start();

  
  for (int kz = 0; kz < output.Nz(); kz ++) {
    if (kz + local_0_start > global_z_size/2)
      qz = (-global_z_size + kz + local_0_start ) * differential[2];
    else
      qz = ( kz + local_0_start ) * differential[2];
    for (int jy = 0; jy < output.Ny(); jy ++ ) {
      if (jy > global_y_size/2)
	qy = (-global_y_size + jy ) * differential[1];
      else
	qy =  jy * differential[1];
      for (int ix = 0; ix < output.Nx(); ix ++ ) {
	qx = ix * differential[0];
	
	output(ix,jy,kz) = func(qx,qy,qz,object);
	
      }
    }
  }
  return;
}



void c2c_3d_apply_function(std::complex<double>
			   (*func)(double,double,double,void*),
			   void *object,fftwArr::c2c_3D & output,
			   const std::array<double,3> & differential,
			   const std::array<double,3> & origin)
{
  double qx,qy,qz;
  int global_z_size = output.global_Nz();
  int global_y_size = output.global_Ny();
  int global_x_size = output.global_Nx();
  int local_0_start = output.get_local0start();

  
  for (int kz = 0; kz < output.Nz(); kz ++) {
    if (kz + local_0_start > global_z_size/2)
      qz = (-global_z_size + kz + local_0_start ) * differential[2];
    else
      qz = ( kz + local_0_start ) * differential[2];
    for (int jy = 0; jy < output.Ny(); jy ++ ) {
      if (jy > global_y_size/2)
	qy = (-global_y_size + jy ) * differential[1];
      else
	qy =  jy * differential[1];
      for (int ix = 0; ix < output.Nx(); ix ++ ) {
	if (ix > global_x_size/2)
	  qx = (-global_x_size + ix ) * differential[0];
	else
	  qx = ix * differential[0];
	
	output(ix,jy,kz) = func(qx,qy,qz,object);
	
      }
    }
  }
  
  return;
}
  

double func_phi(double x,double y,double z,void *object)
{
  double *real = static_cast<double*>(object);
  return sin(x)*cos(y)*sin(2*z)*(*real);
}


std::complex<double> func_ft_theta(double x,double y,double z,
				   void *object)
{
  double *real = static_cast<double*>(object);
  return sin(x)*cos(y)*sin(2*z)*(*real);
}



std::complex<double> func_complex_psi(double x,double y,double z,
				      void *object)
{
  double *real = static_cast<double*>(object);
  return sin(x)*cos(y)*sin(2*z)*(*real);
}


