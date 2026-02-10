#ifndef FFTWMPI_CONJ_NODES_HPP
#define FFTWMPI_CONJ_NODES_HPP

#include "fftw_arr.hpp"

namespace fftwArr {
struct Node {
  int proc;
  ptrdiff_t bounds[2];
};


class ConjNodes
{
public:


  ConjNodes(ptrdiff_t,ptrdiff_t,MPI_Comm);
  ~ConjNodes();
  std::string print_details() const;


  std::vector<Node> send_nodes, recv_nodes;
  

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


  std::vector<ptrdiff_t> list_of_local_0_starts;


  MPI_Datatype MPI_NodeType;


  std::vector<std::array<ptrdiff_t,2>> list_of_left_bounds;
  std::vector<std::array<ptrdiff_t,2>> list_of_right_bounds;

  std::vector<std::vector<Node>> global_node_sends;
  std::vector<std::vector<Node>> global_node_recvs;  

  

  void set_global_bounds();
  void set_left_and_right();
  void set_local_bounds();

  void share_local_to_global();
  void share_local_lefts();
  void share_local_rights();  
  void share_local_0_starts();

  
  void decide_left_sends_recvs();
  void init_left_sends_recvs(std::array<ptrdiff_t,2> &,
			     std::array<ptrdiff_t,2> &);



  void share_left_sends_recvs();
  void share_global_list_to_processors(std::vector<std::vector<int>> &,
				       const std::vector<int> &);
  void share_global_list_to_processors(std::vector<std::vector<Node>> &,
				       const std::vector<Node> &);


  void update_right_sends_recvs();
  void finit_right_sends_recvs();
  std::array<ptrdiff_t,2> get_vector_from_nodes(const std::vector<Node> &) const;
  
  void set_MPI_NodeType();
  

};




};



#endif
