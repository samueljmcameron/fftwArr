# Sine wave example

This example discretises the wave $\phi(x,y,z)=\sin(x)\cos(y)\sin(2z)$
in the region $x\in[0,2\pi)$, $y\in[0,2\pi)$, $z\in[0,2\pi)$ and
approximates its derivatives using FFTW3 (MPI) module. Discretisation
is chosen so that there are 40 evenly spaced points in the $x$ direction,
34 in the $y$ direction, and 20 in the $z$ direction. The outer axis
is chosen to be the $z$ axis, and FFTW3 splits this axis among
all MPI processes. Output is written as csv (text) files
with the names ``output_%.txt'' where ``%'' is replaced by
the processor number. The python script ``plot.py'' can be
used to visualise the output.

To build/run this example from this directory do:

mkdir build
cd build
cmake -D CMAKE_PREFIX_PATH=<whereverfftwArrisinstall> ../