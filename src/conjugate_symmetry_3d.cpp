#include "array3d.hpp"


using namespace fftwArr;

ConjugateSymmetry3D::ConjugateSymmetry3D()
  : nblocks(4), ft_array(nullptr)
{


  int nprocs = ft_array->get_nprocs();
  int me = ft->array->get_me();
  MPI_Comm world = ft_array->get_world();

  
}


void ConjugateSymmetry3D::setup()
{


  std::vector<int> split_sizes = ft_array->split_sizes();

  check_splits(split_sizes);

  int first_split_size = split_sizes.at(0);
  int last_split_size = split_sizes.at(nprocs -1);

  if (first_split_size == last_split_size) {


  }



  
  
}



void ConjugateSymmetry3D::check_splits(const std::vector<int> &splits)
{
  int local_err,global_err;
  std::string error_message;

  local_err = 0;
  
  for (int i = 1; i < nprocs; i++) {
    if (split_sizes.at(nprocs-i) == 0) {
      error_message = "Conjugate symmetry 3D cannot have processors with zero data.";
      local_err = 1;
    } else if (split_sizes.at(0) < split_sizes.at(i)) {
      error_message = "Final axis size of process "
	+ std::to_string(me)
	+ " is larger than block size of process 0";
      local_err = 1;
    } else if (split_sizes.at(i-1) != split_sizes.at(i)
	       && i != nprocs -1) {
      error_message = "Final axis size of process "
	+ std::to_string(me)
	+ " must be the same as final axis size of process"
	+ std::to_string(me-1);
      local_err = 1;
    }
  }

  MPI_Allreduce(&local_err, &global_err, 1, MPI_INT, MPI_SUM,
		world);

  if (global_err > 0)
    throw std::runtime_error(error_message);

}
