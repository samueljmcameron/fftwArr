#include <vector>
#include <array>
#include <string>
#include <mpi.h>
#include <cstddef>
#include <cstdint>

#include <iostream>

#include "conj_nodes.hpp"


using namespace fftwArr;



ConjNodes::ConjNodes
(ptrdiff_t local_axis_size, ptrdiff_t local_0_start,MPI_Comm world)
  : local_axis_size(local_axis_size),local_0_start(local_0_start),
    world(world),left(0),right(0),global_left_bounds{},
    global_right_bounds{},left_bounds{},right_bounds{}
{

  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);

  set_MPI_NodeType();


  MPI_Allreduce(&local_axis_size, &global_axis_size, 1, MPI_AINT, MPI_SUM, world);

  set_global_bounds();


  set_left_and_right();
  
  set_local_bounds();
    

  share_local_to_global();

  decide_left_sends_recvs();

  share_left_sends_recvs();
  update_right_sends_recvs();
  
  
  
}


ConjNodes::~ConjNodes()
{
  MPI_Type_free(&MPI_NodeType);
}





void ConjNodes::set_global_bounds()
/*
  determine where the split between left and right is globally
 */
{

  // left indices as in left of the middle value (which
  //  is global_axis_size/2)

  global_left_bounds.at(0) = 1;
  global_left_bounds.at(1) = (global_axis_size+1)/2;

  

  // right indices as in right of the middle value (which
  //  is global_axis_size/2)

  global_right_bounds.at(0) = global_axis_size/2 + 1;
  global_right_bounds.at(1) = global_axis_size;

  return;
  
}

void ConjNodes::set_left_and_right()
/*
  set left = 1 if processor is left of the global split
  set right = 1 if processor is right of the global split
  
  (can be both)
  
*/
{

  
  // sanity check example of next two if statements:
  //
  //    If global_axis_size = 6, you want processors with global
  //    indices 0,1,2 saying that they have indices on the left
  //    of global_axis_size//2 and processors with global indices 4,5,6
  //    saying that they have indices on to the right of
  //    global_axis_size//2. 
  //
  //    If global_axis_size = 5, you want processors with global
  //    indices 0,1,2 saying that they have indices on the left
  //    of global_axis_size//2 and processors with global indices 3,4
  //    saying that they have indices on to the right of
  //    global_axis_size//2. 
  //
  

  
  if (local_0_start < global_left_bounds.at(1))
    left = 1; 
  
  if (local_axis_size + local_0_start > global_right_bounds.at(0))
    right = 1;

  return;
}





void ConjNodes::set_local_bounds()
/*
  Find the left and right index bounds for the local processor.
  The lower bound is always included in the for loop, while the
  upper bound is always one greater than the for loop upper bound

  e.g. for (int i = left_bounds.at(0); i < left_bounds.at(1); i++)

  
  Do not include the global index 0 (or the global index
  global_axis_size/2 if global_axis_size % 2 == 0) since those
  two indices are not to be shared between processors.
  
 */  
{

  
  if (right && left) {

    if (me == 0)
      left_bounds.at(0) = 1;
    else
      left_bounds.at(0) = 0;
    left_bounds.at(1) = global_left_bounds.at(1)-local_0_start;


    right_bounds.at(0) = global_right_bounds.at(0)-local_0_start;
    right_bounds.at(1) = local_axis_size;



  } else if (left) {

    if (me == 0)
      left_bounds.at(0) = 1;
    else
      left_bounds.at(0) = 0;

    left_bounds.at(1) = local_axis_size;
    
    // if the processor's largest index == global_axis_size/2 (no
    // integer division), then the left bound needs to shift down by
    // one
    if (global_axis_size % 2 == 0
	&& local_axis_size + local_0_start == global_right_bounds.at(0)) {
      std::cout << "MADE IT HERE ON PROCESSOR " << me << std::endl;
      left_bounds.at(1) -= 1;
    }


    // set right bounds so that it will not trigger a for loop
    right_bounds.at(0) = local_axis_size;
    right_bounds.at(1) = local_axis_size;

    

  } else if (right) {

    // set left bounds so that it will not trigger a for loop
    left_bounds.at(0) = 0;
    left_bounds.at(1) = 0;


    
    right_bounds.at(0) = 0;

    
    // if the processor's smallest index == global_axis_size/2 (no
    // integer division), then the right bound needs to shift up by
    // one
    
    if (global_axis_size % 2 == 0
	&& local_0_start == global_left_bounds.at(1))
      right_bounds.at(1) += 1;

    
    right_bounds.at(1) = local_axis_size;

  }
  return;
}




void ConjNodes::decide_left_sends_recvs()
{

  // indices for sending and receiving from the current
  // processor. 
  std::array<ptrdiff_t,2> send_bounds,recv_bounds;

  
  // initially, find the send_bounds and recv_bounds for the left side
  init_left_sends_recvs(send_bounds,recv_bounds);

  
  // determine the send and receive 

  
  std::array<ptrdiff_t,2> send_to;
  std::array<ptrdiff_t,2> recv_from;


  
  if (left) {


    // find which processors to send/recv using symmetry
    // of index -> global_axis_size- index

    auto & sb_tmp = send_bounds;
    auto & rb_tmp = recv_bounds;

    send_to.at(0) = global_axis_size - sb_tmp.at(1)-local_0_start + 1 ;
    send_to.at(1) = global_axis_size - sb_tmp.at(0)-local_0_start + 1;

    recv_from.at(0) = global_axis_size - rb_tmp.at(1)-local_0_start + 1;
    recv_from.at(1) = global_axis_size - rb_tmp.at(0)-local_0_start + 1;


    if (send_to.at(1) - send_to.at(0) > 0) {
      for (int p = 0; p < nprocs; p++) {
	
	auto & right_tmp = list_of_right_bounds.at(p);
	
	auto tmp_l0 = 0;//list_of_local_0_starts.at(p);
	
	auto low = right_tmp.at(0);
	auto hi = right_tmp.at(1);
	
	ptrdiff_t first,last;
	
	if (hi - low > 0 && hi > send_to.at(0) && low < send_to.at(1)) {

	  first = low < send_to.at(0) ? send_to.at(0) : low;
	  
	  last = hi > send_to.at(1) ? send_to.at(1) : hi;
	  
	  Node node;
	  node.proc = p;
	  node.bounds[0] = first;
	  node.bounds[1] = last;
	  send_nodes.push_back(node);

	  
	}
      
	
      }
    }

    if (recv_from.at(1) - recv_from.at(0) > 0) {
      for (int p = 0; p < nprocs; p++) {
	
	auto & right_tmp = list_of_right_bounds.at(p);
	
	auto low = right_tmp.at(0);
	auto hi = right_tmp.at(1);

	if (me == 1) {
	  if (hi <= recv_from.at(0))
	    std::cout << "proc: " << p << " has "
		      << low << "," << hi
		      << " and " << recv_from.at(0) << std::endl;
	}
	
	ptrdiff_t first,last;
	
	if (hi - low > 0 && hi > recv_from.at(0) && low < recv_from.at(1)) {
	  
	  first = low < recv_from.at(0) ? recv_from.at(0) : low;
	  
	  last = hi > recv_from.at(1) ? recv_from.at(1) : hi;

	  Node node;
	  node.proc = p;
	  node.bounds[0] = first;
	  node.bounds[1] = last;
	  recv_nodes.push_back(node);
	  

	}
      
	
      }


    }
  }
  
  
}


void ConjNodes
::init_left_sends_recvs(std::array<ptrdiff_t,2> &send_bounds,
			std::array<ptrdiff_t,2> &recv_bounds)
{

  int splitter;
  
  if (right && left) {
    
    // split the left side into send and receive
    splitter = (left_bounds.at(1)-left_bounds.at(0))/2;
    
    
  } else if (left) {
    
    splitter = left_bounds.at(1)/2;


  }

  if (left) {

    send_bounds.at(0) = left_bounds.at(0);
    send_bounds.at(1) = left_bounds.at(0) + splitter;
    recv_bounds.at(0) = send_bounds.at(1);
    recv_bounds.at(1) = left_bounds.at(1);

  }
    
  return;
}



  
void ConjNodes::share_left_sends_recvs()
{
  share_global_list_to_processors(global_node_sends,
				  send_nodes);

  share_global_list_to_processors(global_node_recvs,
				  recv_nodes);
}  



void ConjNodes::update_right_sends_recvs()
{

  
  if (right) {
    
    for (int proc = me-1; proc >= 0; proc--) {
      
      for (auto node : global_node_sends.at(proc))
	
	if (node.proc == me) {

	  node.proc = proc;
	  auto tmp0 = node.bounds[0];
	  auto tmp1 = node.bounds[1];
	  node.bounds[0] = global_axis_size-tmp1+1;
	  node.bounds[1] = global_axis_size-tmp0+1;
	  recv_nodes.push_back(node);

	}
      
      
      for (auto node : global_node_recvs.at(proc))
	
	if (node.proc == me) {

	  node.proc = proc;
	  auto tmp0 = node.bounds[0];
	  auto tmp1 = node.bounds[1];
	  node.bounds[0] = global_axis_size-tmp1+1;
	  node.bounds[1] = global_axis_size-tmp0+1;
	  send_nodes.push_back(node);

	}
    }
  }

}

/*
std::array<ptrdiff_t,2> ConjNodes
::get_vector_from_nodes(const std::vector<Node> & nodes) const
{

  ptrdiff_t low = PTRDIFF_MAX;
  ptrdiff_t high = 0;


  for (const auto &node : nodes) {
    if (node.bounds[0] < low)
      low = node.bounds[0];
    if (node.bounds[1] > high)
      high = node.bounds[1];
  }

  return {low-1,high-1};

}
*/



void ConjNodes
::share_global_list_to_processors(std::vector<std::vector<Node>> 
				  &global_list_to_processors,
				  const std::vector<Node> &to_processors)
{


  // make sure the global list has nproc elements
  global_list_to_processors.resize(nprocs);


  // create array to store the number of nodes for each local processor
  std::vector<int> nodes_per_proc(nprocs);

  // get local number of nodes
  const int local_node_number = to_processors.size();
  

  // communicate the local number of nodes to to_psizes
  MPI_Allgather(&local_node_number,1,MPI_INT,
		nodes_per_proc.data(),1,MPI_INT,world);




  // get the displacements of each node
  std::vector<int> displacements;

  // add up all the total number of nodes across all processors
  int sum = 0;
  for (int p = 0; p < nprocs; p++) {
    displacements.push_back(sum);
    sum += nodes_per_proc.at(p);
  }

  // temporary array to store all entries of nodes in order in a flattened
  // list (do this vs direct send to global list to avoid mpi complexities)
  std::vector<Node> tmparray(sum);
  

  MPI_Allgatherv(to_processors.data(),local_node_number,
		 MPI_NodeType,tmparray.data(),nodes_per_proc.data(),
		 displacements.data(),MPI_NodeType,world);
  


  // transfer data from tmparray to global list
  for (int p = 0; p < nprocs; p++) {
    global_list_to_processors.at(p).resize(nodes_per_proc.at(p));
    
    for (int j = 0; j < nodes_per_proc.at(p); j++)
      
      global_list_to_processors.at(p).at(j)
	= tmparray.at(j+displacements.at(p));
    
  }
  

  return;
  
}

void ConjNodes::share_local_to_global()
{
  share_local_0_starts();
  share_local_lefts();
  share_local_rights();
}


void ConjNodes::share_local_lefts()
{


  list_of_left_bounds.resize(nprocs);


  MPI_Allgather(&left_bounds[0],2,MPI_AINT,
		list_of_left_bounds.data(),2,MPI_AINT,
		world);

  for (int p = 0; p < nprocs; p++)
    for (auto & item : list_of_left_bounds.at(p) )
      item += list_of_local_0_starts.at(p);
  return;
  
}


void ConjNodes::share_local_rights()
{


  list_of_right_bounds.resize(nprocs);


  MPI_Allgather(&right_bounds[0],2,MPI_AINT,
		list_of_right_bounds.data(),2,MPI_AINT,
		world);

  for (int p = 0; p < nprocs; p++)
    for (auto & item : list_of_right_bounds.at(p) )
      item += list_of_local_0_starts.at(p);
  
  return;
  
}




void ConjNodes::share_local_0_starts()
{


  list_of_local_0_starts.resize(nprocs);

  MPI_Allgather(&local_0_start,1,MPI_AINT,
	     list_of_local_0_starts.data(),1,MPI_AINT,
	     world);
  return;
}

  
std::string ConjNodes::print_details() const
{

  std::string output;
  output = "\nProcessor " + std::to_string(me)
    + std::string(":\n----------------\n");
  
  if (left) {
    output += "LEFT\n"; 
    output += "min index: "
      + std::to_string(left_bounds.at(0) + local_0_start) + std::string("\n");
    output += "max index: "
      + std::to_string(left_bounds.at(1) + local_0_start - 1) + std::string("\n");
  }
  if (right) {
    output += "RIGHT\n";
    output += "min index: "
      + std::to_string(right_bounds.at(0) + local_0_start) + std::string("\n");
    output += "max index: "
      + std::to_string(right_bounds.at(1) + local_0_start - 1) + std::string("\n");
  }
  
  output += "SEND_TO_PROCESSORS:\n";
  
  output += "(proc,first,last)\n";
  
  for (auto & item : send_nodes) {
    output += "(";
    output += std::to_string(item.proc) + ",";
    output += std::to_string(item.bounds[0]) + ",";
    output += std::to_string(item.bounds[1]) + ")\n";
    
  }

  
  output += "RECV_FROM_PROCESSORS:\n";
  
  output += "(proc,first,last)\n";
  
  
  for (auto & item : recv_nodes) {
    output += "(";
    output += std::to_string(item.proc) + ",";
    output += std::to_string(item.bounds[0]) + ",";
    output += std::to_string(item.bounds[1]) + ")\n";
    
  }
 
  return output;

}

void ConjNodes::set_MPI_NodeType()
{

  MPI_Datatype tmptype;
  MPI_Datatype oldtypes[2];
  int blockcounts[2];
  MPI_Aint offsets[2];
  MPI_Status status;

  offsets[0] = offsetof(Node,proc);
  oldtypes[0] = MPI_INT;
  blockcounts[0] = 1;


  offsets[1] = offsetof(Node,bounds);
  oldtypes[1] = MPI_AINT;
  blockcounts[1] = 2;

  MPI_Type_create_struct(2,blockcounts,offsets,oldtypes,&tmptype);

  MPI_Aint lb, extent;
  MPI_Type_get_extent(tmptype, &lb, &extent);

  MPI_Type_create_resized(tmptype, lb, extent, &MPI_NodeType);

  
  MPI_Type_commit(&MPI_NodeType);

  MPI_Type_free(&tmptype);

  
}
