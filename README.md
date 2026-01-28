# fftwArr

Array interface for mpi-enabled Fastest Fourier Transform in the West (FFTW) C
subroutine library (i.e. fftw3_mpi). This library defines several of the data
structures used in FFTW. Namely, three classes 

  fftwArr::r2c_2D  // (real-to-complex)
  fftwArr::c2r_2D  // (complex-to-real)
  fftwArr::c2c_2D  // (complex-to-complex)

whose main member data structure is a two-dimensional array, and three classes

  fftwArr::r2c_3D  // (real-to-complex)
  fftwArr::c2r_3D  // (complex-to-real)
  fftwArr::c2c_3D  // (complex-to-complex)

whose main member data structure is a three-dimensional array. The phrases
"real-to-complex", "complex-to-real", and "complex-to-complex" mean very specific
things in terms of how these arrays are stored in memory, consistent with FFTW.
The details are in the FFTW doc so I'd recommend reading that (which surely you will
if you are interested in using this library), but the main thing is that the data
storate in these arrays is generally not contiguous in memory due to some (necessary)
quirks of the FFTW library. These quirks are a strong motivator for me making this
library. Otherwise, in the above class declarations, the first word (e.g. real
in "real-to-complex") indicates whether the array contains real (double) or
complex (std::complex<double>) data.


Now, assuming you're happy with the above (and if not please read the FFTW docs),
how can we use these classes practically. Below is a very detailed example of how
to use this library best with FFTW. It should also hopefully clear up any remaining
questions you might have.

--------------------------------------------------------------------------------
EXAMPLE START


Suppose you have a (real, 3D) scalar field PHI(x,y,z), which you want to
Fourier transform, and which you've decided you want to use the real-to-complex
transform available in the FFTW library to do it.
You can use the fftwArr library to create an object to store
this field in memory,

  fftwArr::r2c_3D phi(world,"phi",Nx,Ny,Nz);

where the arguments to the constructor are

  world - MPI_Comm datatype (typically just MPI_COMM_WORLD)
  "phi" - string which states the name of the array which can be accessed
          using the get_name() class method which is useful for e.g. saving
          (in this case, we've creatively named our array "phi" but could have
	  used any string).
  Nx    - the (integer) number of grid points in the x direction
  Ny    - the (integer) number of grid points in the y direction
  Nz    - the (integer) number of grid points in the z direction.


The array is then spread across all processors, so that each processor has a block
as shown below (for a 3D array created across p processors):

  y
  |  x
  | /
  |/
   ------ z
  
        _______    _______    _______           _______
       /      /|  /      /|  /      /|         /      /|
  Nx  /      / | /      / | /      / |        /      / | 
     /      /  |/      /  |/      /  |       /      /  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |  ...  |      |  |
     |      |  ||      |  ||      |  |       |      |  |
 Ny  |      |  ||      |  ||      |  |       |      |  |
     |      |  ||      |  ||      |  |       |      |  |
     |      |  /|      |  /|      |  /       |      |  /
     |      | / |      | / |      | /        |      | /
     |______|/  |______|/  |______|/         |______|/

       Nz_1       Nz_1       Nz_2       ...    Nz_p


Note that, consistent with FFTW, each block may have a different number of z
grid points. So generally Nz_1 != Nz_2, etc. All the same caveats apply as mentioned
in the FFTW docs about best to choose non-prime global number of grid points in each
dimension (so probably don't choose Nz = 13 or Nx = 23 if you can help it).



Since you want to Fourier Transform phi, you need to create a complex-to-real array
to store the Fourier Transform. Now, I'm going to take advantage of the fact that
FFTW also says you can save some time by allowing the Fourier Transform to deal
with an array whos two outer most indices are transposed (see the FFTW_MPI_TRANSPOSED_OUT
flag in FFTW). In this case, I will write


  fftwArr::c2r_3D ft_phi(MPI_COMM_WORLD,"ft_phi",Nx,Nz,Ny);
  //                                                ^  ^
  //                                                |  |
  //					  Ny and Nz are swapped here
  //                                      since we are going to use the
  //                                      FFTW_MPI_TRANSPOSED_OUT option
  //                                      in the FFTW library.


Just to reiterate, you can just as easily not use the transposed option, and
in that case you'd just list your grid points as Nx,Ny,Nz just as you did for
the real array phi.

The next thing you need to do is create a fftw_plan (see FFTW docs for details).
For the arrays we have so far, you can simply write

  fftw_plan forward_phi
    = fftw_mpi_plan_dft_r2c_3d(Nz,Ny,Nx,phi.data(),
			       reinterpret_cast<fftw_complex*>
			       (ft_phi.data()),
			       MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT);


Now, it's important to note a few things here.

1) Unfortunately have an inconsistency of notation between FFTW and fftwArr.
   In the former the arrays are assumed to be row-major (C) order, but in fftwArr
   we have decided to use column-major (fortran) order to permit easy writing to
   VTK files. That means that we must list Nx, Ny, and Nz in what appears to be
   the wrong order in FFTW's fftw_mpi_plan_dft_r2c_3d function as shown above.
   This is an unfortunate conflict I've decided to live with for reasons discussed
   below.
2) For both the (real) data in the phi class and the (complex) data in the ft_phi
   class, the array can be accessed using the data() member function. Unfortunately
   there is another small inconvenience here in which we must explicitly recast
   the data from std::complex<double> to fftw_complex, but it's not a huge deal...
3) For other details on the above function call, see FFTW docs.


Then, you can set the elements of the array (on your local processor)
using for loops by using the fact that

  phi(ix,jy,kz)

allows read/write access the value of phi at x argument indexed by ix, y index
by iy, and z value indexed by iz. So

for (int kz = 0; kz < phi.Nz(); kz ++ ) {
  z = ( kz + phi.get_local0start() ) * dz + z_origin;
  for (int jy = 0; jy < phi.Ny(); jy ++ ) {
    y = jy * dy + y_origin;
    for (int ix = 0; ix < phi.Nx(); ix ++ ) {
      x = ix * dx + x_origin;
      phi(ix,jy,kz) = PHI(x,y,z); // PHI(x,y,z) is some function like sin(x)*4y + z
    }
  }
}

A couple things can be said from the above (triple) for loop.

1) As mentioned above, the fftwArr is in column-major (fortran) order. The reason for
   this (alluded to above) is that I wanted to write the arrays out to VTK ImageData,
   and VTK ImageData requires that the x index varyies the quickest to be able to
   visualised correctly in e.g. Paraview. Therefore, I was left with the choice of
   indexing in a reverse order (where the z index is first) to maintain the appearance
   of row-major (C) order, or just bite the bullet and go with column-major order.
   I opted for the latter (after an initial period of trying the former).

2) The local sizes of each dimension are returned from the class methods Nx(), Ny(),
   Nz(). This is useful for looping over the arrays. Remember that the integers
   returned by Nx(), Ny(), and  Nz() are not equal to their global counterparts
   in general. Generically, Nz() != Nz unless there is only one processor. Specifically,
   Nx() can be equal to Nx (for r2c arrays) but is not always (for c2r arrays). So
   just always treat them as different entities.


3) The value which z takes must account for the splitting of the data across processors.
   The integer returned by the get_local0start() method provides the offset of the
   current processor which allows you to determine globally which z value you should
   be at.


Now that the data is stored in the phi class, one can just perform the transform using
FFTW's standard

  fftw_execute(forward_phi);

And that's it! You've done a Fourier Transform on real, 3D scalar function. There are
(compileable) examples of this type of thing in the examples folder which you'll
need to look at for the coding details, but overall it's not much more difficult
conceptually than this.

End Example
--------------------------------------------------------------------------------

The classes of this library also provide member functions which let you read/write
data to/from files (in binary format). The binary format is compatible with
VTK's appended binary data so that is useful. Briefly, if one instantiates 

  fftwArr::r2c_3D phi(world,"phi",Nx,Ny,Nz);

then to save the data (on the local processor) to a file, do the following:

  std::fstream writefile;
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);

  std::string filename = std::string("filename_p") + std::to_string(me);

  writefile.open(filename,
		 std::fstream::binary | std::fstream::out);

  phi.write_to_binary(writefile);


One can then read this data back in later using

  phi.read_from_binary(readfile);