import numpy as np

x=np.load("box.npy")
x=x.astype('float64')
nf=x.shape[0]
np.save("box.npy",x.reshape((nf,6)))


x=np.load("pos.npy")
x=x.astype('float64')
np.save("pos.npy",x.reshape((nf,-1)))


x=np.load("energy.npy")
x=x.astype('float64')
np.save("energy.npy",x)


x=np.load("force.npy")
x=x.astype('float64')
np.save("force.npy",x.reshape((nf,-1)))
breakpoint()
