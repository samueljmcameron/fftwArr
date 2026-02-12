#include <vector>
#include <array>
#include <string>
#include <mpi.h>
#include <cstddef>
#include <cstdint>

#include "conjugate_symmetry_3d.hpp"


using namespace fftwArr;


ConjugateSymmetry3D::ConjugateSymmetry3D(c2r_3D *ft_array)
  : ft_array(ft_array)
{



  ConjNodes conj_nodes(ft_array->size_axis2(),ft_array->get_local0start(),
		       ft_array->get_world());

  send_nodes = conj_nodes.send_nodes;
  recv_nodes = conj_nodes.recv_nodes;

  envelope.resize(ft_array->size_axis1(),ft_array->size_axis2());
  
}



void ConjugateSymmetry3D::transfer_to_envelope( ptrdiff_t ix)
{
  ptrdiff_t lo, hi;
  for (auto & node : send_nodes) {
    lo = node.my_bounds[0];
    hi = node.my_bounds[1];

    
    for (ptrdiff_t kz = lo; kz < hi; kz ++)
      for (ptrdiff_t jy = 0; jy < ft_array->size_axis1(); jy ++ )
	envelope(jy,kz) = (*ft_array)(ix,jy,kz);
  }

  return;
}


void ConjugateSymmetry3D
::populate_sends(std::complex<double> (*func)(double,double,double,void *),
		 void *params,ptrdiff_t ix,
		 const std::array<double,3> & differential,
		 const std::array<double,3> & origin)
{


  ptrdiff_t local_0_start = ft_array->get_local0start();
  ptrdiff_t global_y_size = ft_array->global_Ny();
  ptrdiff_t global_z_size = ft_array->global_Nz();
  
  ptrdiff_t lo, hi;
  double qx,qy,qz;

  qx = ix * differential[0];
  
  
  for (auto & node : send_nodes) {
    lo = node.my_bounds[0] - ft_array->get_local0start();
    hi = node.my_bounds[1] - ft_array->get_local0start();

    
    if (ft_array->is_transposed()) {

      for (ptrdiff_t jy = lo; jy < hi; jy ++) {
	if (jy + local_0_start > global_y_size/2)
	  qy = (-global_y_size + jy + local_0_start ) * differential[1];
	else
	  qy = ( jy + local_0_start ) * differential[1];
	for (ptrdiff_t kz = 0; kz < ft_array->size_axis1(); kz ++ ) {
	  if (kz > global_z_size/2)
	    qz = (-global_z_size + kz ) * differential[2];
	  else
	    qz =  kz * differential[2];

	    
	  (*ft_array)(ix,kz,jy) = func(qx,qy,qz,params);

	}

      }
      
    } else {

      for (ptrdiff_t kz = lo; kz < hi; kz ++) {
	if (kz + local_0_start > global_z_size/2)
	  qz = (-global_z_size + kz + local_0_start ) * differential[2];
	else
	  qz = ( kz + local_0_start ) * differential[2];
	for (ptrdiff_t jy = 0; jy < ft_array->size_axis1(); jy ++ ) {
	  if (jy > global_y_size/2)
	    qy = (-global_y_size + jy ) * differential[1];
	  else
	    qy =  jy * differential[1];
	    
	  (*ft_array)(ix,jy,kz) = func(qx,qy,qz,params);
	}

      }

      
    }
    
      
  }

}

/*

void ConjugateSymmetry3D::fill_envelopes(ptrdiff_t nx)
{


  ptrdiff_t lo, hi;


  
  for (auto & node : send_nodes) {
    lo = node.my_bounds[0];
    hi = node.my_bounds[1];


    for (ptrdiff_t nz = lo; nz < hi; nz++)
      for (ptrdiff_t ny = 0; ny < ft_array->size_axis1(); ny++)
      
	envelope(ny,nz) = (*ft_array)(nx,ny,nz);

  }
  
  


}
*/
