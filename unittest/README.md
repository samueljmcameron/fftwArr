This directory contains the following:


1) A testing utilities folder which holds source code for a small
 library (to be compiled and installed somewhere prior to running
 any of the tests below).
 -- to install this small library, do the following:

    cd utils
    mkdir build; cd build;
    cmake -D CMAKE_PREFIX_PATH=<path to fftwArr> -D CMAKE_INSTALL_PREFIX=<where to install> ../
    make
    make install
    
2) All tests for the fftwArr arrays (2D and 3D) which are in
   subdirectories which begin with "test-".
   -- to run test-blah, one must FIRST install the utils library
   mentioned in item 1 above and then do the following:
   
      cd test-blah
      mkdir build; cd build;
      cmake -D CMAKE_PREFIX_PATH="<path to fftwArr>;<path to fftwArrTestingUtils>" ../
      make
      ./test-blah
      mpiexec -np <number of processors> ./test-blah
   
