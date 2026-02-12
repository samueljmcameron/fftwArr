#ifndef FFTWMPI_CONJUGATE_SYMMETRY_3D_NODES_HPP
#define FFTWMPI_CONJUGATE_SYMMETRY_3D_NODES_HPP

#include "conj_nodes.hpp"
#include "arrays.hpp"
#include "smatrix.hpp"

namespace fftwArr {

class ConjugateSymmetry3D
{
public:


  ConjugateSymmetry3D(c2r_3D *);

  void transfer_to_envelope( ptrdiff_t ix);

  void populate_sends(std::complex<double> (*func)(double,double,double,void *),
		      void *,ptrdiff_t,const std::array<double,3> &,
		      const std::array<double,3> & origin = {0,0,0});
  

  std::vector<Node> send_nodes, recv_nodes;

private:

  c2r_3D *ft_array;

  ptrdiff_t local_0_start;

  sMatrix<std::complex<double>> envelope;
  

};




};



#endif
