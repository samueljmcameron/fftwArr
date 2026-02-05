#include <vector>
#include <array>
#include <string>
#include <mpi.h>

#include <iostream>

#include "conjugate_symmetry_base.hpp"


using namespace fftwArr;



template < enum Transform rOc,typename T>
ConjugateSymmetryBase<rOc,T>::ConjugateSymmetryBase
(ptrdiff_t local_axis_size, ptrdiff_t local_0_start,MPI_Comm world)
  : local_axis_size(local_axis_size),local_0_start(local_0_start),
    world(world),left(0),right(0),global_left_bounds{},
    global_right_bounds{},left_bounds{},right_bounds{},
    send_bounds{},recv_bounds{}
{


  MPI_Comm_size(world,&nprocs);
  MPI_Comm_rank(world,&me);


  MPI_Allreduce(&local_axis_size, &global_axis_size, 1, MPI_AINT, MPI_SUM, world);

  set_global_bounds();

  set_left_and_right();
  set_local_bounds();


  
    

  share_local_0_starts();

  share_local_lefts();
  share_local_rights();
  
  set_sends_recvs();

  
  
}


template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::init_left_sends_recvs()
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


template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::set_local_bounds()
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


    // set right bounds so that it will not trigger a for loop
    right_bounds.at(0) = local_axis_size;
    right_bounds.at(1) = local_axis_size;

    

  } else if (right) {

    // set left bounds so that it will not trigger a for loop
    left_bounds.at(0) = 0;
    left_bounds.at(1) = 0;
    
    right_bounds.at(0) = 0;
    right_bounds.at(1) = local_axis_size;

  }
  return;
}

template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::set_left_and_right()
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

template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::set_global_bounds()
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

template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::set_sends_recvs()
{


  // initially, find the send_bounds and recv_bounds for the left side
  init_left_sends_recvs();

  
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


    if (me == 1)
      std::cout << recv_bounds.at(0) << std::endl;

    if (send_to.at(1) - send_to.at(0) > 0) {
      for (int p = 0; p < nprocs; p++) {
	
	auto & right_tmp = list_of_right_bounds.at(p);
	
	auto tmp_l0 = 0;//list_of_local_0_starts.at(p);
	
	auto low = right_tmp.at(0);
	auto hi = right_tmp.at(1);
	
	ptrdiff_t first,last;
	
	if (hi - low > 0 && hi > send_to.at(0) && low < send_to.at(1)) {
	  
	  send_to_processors.push_back(p);
	  
	  first = low < send_to.at(0) ? send_to.at(0) : low;
	  
	  last = hi > send_to.at(1) ? send_to.at(1) : hi;
	  
	  list_of_send_bounds.push_back({first,last});
	  
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
	  
	  recv_from_processors.push_back(p);
	  
	  first = low < recv_from.at(0) ? recv_from.at(0) : low;
	  
	  last = hi > recv_from.at(1) ? recv_from.at(1) : hi;
	  
	  list_of_recv_bounds.push_back({first,last});
	  
	}
      
	
      }


    }
  }

  /*

    WORKING ON THIS NEXT!
  if (right && !left) {

    
    auto & right_tmp = list_of_right_bounds.at(me);

    recv_from.at(0) = global_axis_size - right_tmp.at(1) + 1;
    recv_from.at(1) = global_axis_size - right_tmp.at(0) + 1;
    

    for (int p = me; p >= 0; p--) {
      auto & lb = list_of_left_bounds.at(p);
      
      if (lb.at(1) - lb.at(0) <= 0) continue; // no left on processor p

      



      
    }

  }
  */
}


template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::share_local_lefts()
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


template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::share_local_rights()
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




template < enum Transform rOc,typename T>
void ConjugateSymmetryBase<rOc,T>::share_local_0_starts()
{


  list_of_local_0_starts.resize(nprocs);

  MPI_Allgather(&local_0_start,1,MPI_AINT,
	     list_of_local_0_starts.data(),1,MPI_AINT,
	     world);
  return;
}

  
template < enum Transform rOc,typename T>
std::string ConjugateSymmetryBase<rOc,T>::print_details() const
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

  if (left) {

    output += "SEND\n";
    output += "index min: "
      + std::to_string(send_bounds.at(0) + local_0_start) + std::string("\n");
    output += "index max: "
      + std::to_string(send_bounds.at(1) + local_0_start- 1) + std::string("\n");
	
    output += "RECEIVE\n";
    output += "index min: "
      + std::to_string(recv_bounds.at(0) + local_0_start) + std::string("\n");
    output += "index max: "
      + std::to_string(recv_bounds.at(1) + local_0_start - 1) + std::string("\n");
    
    output += "SEND_TO_PROCESSORS:\n";
    
    output += "(proc,first,last)\n";

    for (int i = 0; i < send_to_processors.size(); i++) {

      output += "(";
      output += std::to_string(send_to_processors.at(i)) ;
      for (auto &si : list_of_send_bounds.at(i))
	output += "," + std::to_string(si);
      output += ")";
      output += "\n";
    }


    output += "RECV_FROM_PROCESSORS:\n";
    
    output += "(proc,first,last)\n";

    for (int i = 0; i < recv_from_processors.size(); i++) {

      output += "(";
      output += std::to_string(recv_from_processors.at(i)) ;
      for (auto &si : list_of_recv_bounds.at(i))
	output += "," + std::to_string(si);
      output += ")";
      output += "\n";
    }

    
  }
  
 
  return output;

}


template class fftwArr::ConjugateSymmetryBase<fftwArr::Transform::C2R,std::complex<double>>;
