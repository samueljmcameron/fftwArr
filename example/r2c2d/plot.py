import numpy as np
import matplotlib.pyplot as plt
import sys


def phi(x,y):

    return np.sin(x)*np.cos(y)


def gradphi(x,y):

    return [np.cos(x)*np.cos(y),-np.sin(x)*np.sin(y)]

nprocs = int(sys.argv[1])
relativeFilePath = sys.argv[2]


Nx = 36
Ny = 20

# list where each item is a dataset from one of the processors
datas = []
# list where each item is the length of the Ny data on each processor
NySplits = []

for proc in range(nprocs):

    fname = relativeFilePath+f"output_{proc}.txt"
    
    datas.append(np.loadtxt(fname,delimiter=',',skiprows=1))

    NySplits.append(datas[-1][:,0].size//(Nx))
    
    assert(Nx*NySplits[-1] == datas[-1][:,0].size)


assert(np.sum(NySplits) == Ny)

# number of y unique output vectors
dperpoint = len(datas[-1][0,:])
print(dperpoint)

# full data (no longer split by processor) for each output vector
concatenated = np.empty([Nx*Ny,dperpoint],float)


count = 0
for i,ny in enumerate(NySplits):

    concatenated[count:count+ny*Nx,:] = datas[i]
    count += ny*Nx

"""
uxs = concatenated[:Nx,0]
uys = concatenated[:Nx*Ny:Nx,1]
uzs = concatenated[:Nx*Ny*Nz:Nx*Ny,2]
"""

xs = concatenated[:,0]
ys = concatenated[:,1]
print(xs)
print(ys)


errors = {} # store errors between expected output and output generated from example calculation

errors['input'] = phi(xs,ys)-concatenated[:,2]



gps = gradphi(xs,ys)
labels = ['grad_x','grad_y']


XX = xs.reshape(Ny,Nx)
YY = ys.reshape(Ny,Nx)

GXX = gps[0].reshape(Ny,Nx)
GYY = gps[1].reshape(Ny,Nx)

print(len(XX[0,:]),Nx)
plt.plot(XX[0,:],GXX[0,:])
plt.plot(XX[0,:],GYY[0,:])
plt.show()


for i in range(2):
    errors[labels[i]] = gps[i]-concatenated[:,3+i]

tol = 1e-4

for key,val in errors.items():

    print(f'{key} phi is the same within a tolerance of {tol}: '
          f'{np.allclose(val,0,atol=tol)}')
