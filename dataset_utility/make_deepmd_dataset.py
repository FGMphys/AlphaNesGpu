import numpy as np
import sys
import glob as gl
import shutil as sh
from sklearn.model_selection import train_test_split
import os

if sys.argv[1]=='-h' or sys.argv[1][0]=='h':
   print("This program needs 5 files (pos.dat,force.dat,energy.dat,box.dat,type.dat)\n with all information given per frames in a single raw.\n")
   print("Units are not relevant but they must be consistent! Energies must be provided already normalized by the number of particles.")
   print("Box must be of nine components for deepmd. Recast the six components in nine eventually.")
   print("type.dat contains for each raw the number of particles for a given atom type 0,1,.. and so on.")
   print("Now the training programm can deal only with trajectories (and thus force dataset) where particles\n are grouped by atom type.")
   sys.exit()


seed_shuffle=int(sys.argv[1])


os.mkdir("dataset")
os.mkdir("dataset/training")
os.mkdir("dataset/test")
os.mkdir("dataset/training/set.000")
os.mkdir("dataset/test/set.000")

energy=np.loadtxt("energy_original.dat")
nf=energy.shape[0]

dataset=energy.reshape(nf)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/set.000/energy.npy",data_tr)
np.save("dataset/test/set.000/energy.npy",data_ts)



force=np.loadtxt("force_original.dat")
N=int(force.shape[1]/3)
dataset=force.reshape(nf,N*3)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/set.000/force.npy",data_tr)
np.save("dataset/test/set.000/force.npy",data_ts)


pos=np.loadtxt("pos_original.dat")
dataset=pos.reshape(nf,N*3)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/set.000/coord.npy",data_tr)
np.save("dataset/test/set.000/coord.npy",data_ts)

box=np.loadtxt("box_original.dat")
dataset=box.reshape(nf,9)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/set.000/box.npy",data_tr)
np.save("dataset/test/set.000/box.npy",data_ts)

sh.copy("type.raw","dataset/training")
sh.copy("type.raw","dataset/test")
