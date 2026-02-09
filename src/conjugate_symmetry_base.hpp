#ifndef FFTWMPI_CONJUGATE_SYMMETRY_BASE_HPP
#define FFTWMPI_CONJUGATE_SYMMETRY_BASE_HPP

#include "fftw_arr.hpp"

namespace fftwArr {


template < enum Transform rOc,typename T>
class ConjugateSymmetryBase
{


private:

  const ptrdiff_t local_axis_size,local_0_start;
  const MPI_Comm world;


  int me, nprocs;
  
  ptrdiff_t global_axis_size;

  int left,right;


  // first element of bounds arrays are the lowest index,
  // second element is 1 + highest index

  // globally whether a processor occupies space divided
  // by total_axis2/2 (not including this point)
  std::array<ptrdiff_t,2> global_left_bounds, global_right_bounds;


  // indices for left and right sides of current processor.
  // If processor only exists on the left, then
  // right_bounds == {0,0} (and vice versa)

  std::array<ptrdiff_t,2> left_bounds, right_bounds;

  // indices for sending and receiving from the current
  // processor. 
  
  std::array<ptrdiff_t,2> send_bounds,recv_bounds;

  // extra required for the processor which has left and right
  // both (if it exists)

  std::array<ptrdiff_t,2> second_send_bounds,second_recv_bounds;

  std::vector<ptrdiff_t> list_of_local_0_starts;

  std::vector<std::array<ptrdiff_t,2>> list_of_send_bounds;
  std::vector<std::array<ptrdiff_t,2>> list_of_recv_bounds;






  void set_global_bounds();
  void set_left_and_right();
  void set_local_bounds();
  void init_left_sends_recvs();

  void share_local_lefts();
  void share_local_rights();  
  void share_local_sends();
  void share_local_recvs();
  void share_local_0_starts();
  void set_sends_recvs();

  void share_global_list_to_processors(std::vector<std::vector<int>> &,
				       const std::vector<int> &);

public:

  ConjugateSymmetryBase(ptrdiff_t,ptrdiff_t,MPI_Comm);
  std::string print_details() const;

  std::vector<std::array<ptrdiff_t,2>> list_of_left_bounds;
  std::vector<std::array<ptrdiff_t,2>> list_of_right_bounds;

  std::vector<int> send_to_processors,recv_from_processors;

  std::vector<std::vector<int>> global_list_of_send_to_processors;
  std::vector<std::vector<int>> global_list_of_recv_from_processors;
  

};




};



#endif
