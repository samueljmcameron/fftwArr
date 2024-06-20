import numpy as np
import matplotlib.pyplot as plt
import sys


def phi(x,y,z):

    return np.sin(x)*np.cos(y)*np.sin(2*z)


def gradphi(x,y,z):

    return [np.cos(x)*np.cos(y)*np.sin(2*z),-np.sin(x)*np.sin(y)*np.sin(2*z),
            2*np.sin(x)*np.cos(y)*np.cos(2*z)]

nprocs = int(sys.argv[1])
relativeFilePath = sys.argv[2]


Nx = 40
Ny = 34
Nz = 20

datas = []
NzSplits = []

for proc in range(nprocs):

    fname = relativeFilePath+f"output_{proc}.txt"
    
    datas.append(np.loadtxt(fname,delimiter=',',skiprows=1))

    NzSplits.append(datas[-1][:,0].size//(Nx*Ny))
    
    assert(Nx*Ny*NzSplits[-1] == datas[-1][:,0].size)


assert(np.sum(NzSplits) == Nz)

dperpoint = len(datas[-1][0,:])
print(dperpoint)

concatenated = np.empty([Nx*Ny*Nz,dperpoint],float)


count = 0
for i,nz in enumerate(NzSplits):

    concatenated[count:count+nz*Nx*Ny,:] = datas[i]
    count += nz*Nx*Ny

"""
uxs = concatenated[:Nx,0]
uys = concatenated[:Nx*Ny:Nx,1]
uzs = concatenated[:Nx*Ny*Nz:Nx*Ny,2]
"""

xs = concatenated[:,0]
ys = concatenated[:,1]
zs = concatenated[:,2]

errors = {} # store errors between expected output and output generated from example calculation

errors['input'] = phi(xs,ys,zs)-concatenated[:,3]



gps = gradphi(xs,ys,zs)
labels = ['grad_x','grad_y','grad_z']

for i in range(3):
    errors[labels[i]] = gps[i]-concatenated[:,4+i]

tol = 1e-4

for key,val in errors.items():

    print(f'{key} phi is the same within a tolerance of {tol}: '
          f'{np.allclose(val,0,atol=tol)}')
