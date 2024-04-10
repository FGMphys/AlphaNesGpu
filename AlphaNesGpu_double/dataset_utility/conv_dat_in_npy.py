import numpy as np



x=np.loadtxt("box.dat",dtype='float64').reshape((-1,6))
np.save("box.npy",x)
nf=x.shape[0]

x=np.loadtxt("pos.dat",dtype='float64').reshape((nf,-1))
np.save("pos.npy",x)

x=np.loadtxt("force.dat",dtype='float64').reshape((nf,-1))
np.save("force.npy",x)



x=np.loadtxt("energy.dat",dtype='float64').reshape(-1)
np.save("energy.npy",x)
