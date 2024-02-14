import numpy as np
import sys
import glob as gl

from sklearn.model_selection import train_test_split
import os

if sys.argv[1]=='-h' or sys.argv[1][0]=='h':
   print("This program needs 5 files (pos.dat,force.dat,energy.dat,box.dat,type.dat)\n with all information given per frames in a single raw.\n")
   print("Units are not relevant but they must be consistent! Energies must be provided already normalized by the number of particles.")
   print("Box considers only six components. Recast the nine components in six.")
   print("type.dat contains for each raw the number of particles for a given atom type 0,1,.. and so on.")
   print("Now the training programm can deal only with trajectories (and thus force dataset) where particles\n are grouped by atom type.")
   sys.exit()


try:
   seed_shuffle=int(sys.argv[1])
   print("Make_Dataset: Dataset will be shuffled and split in 0.2 and 0.8 for test and train with seed",seed_shuffle)
except:
   sys.exit("Indicate thse seed to compute dataset split")

os.mkdir("dataset")
os.mkdir("dataset/training")
os.mkdir("dataset/test")

energy=np.loadtxt("energy.dat")
nf=energy.shape[0]

dataset=energy.reshape(nf)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/energy.npy",data_tr)
np.save("dataset/test/energy.npy",data_ts)



force=np.loadtxt("force.dat")
N=int(force.shape[1]/3)
dataset=force.reshape(nf,N*3)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/force.npy",data_tr,dtype='float32')
np.save("dataset/test/force.npy",data_ts,dtype='float32')


pos=np.loadtxt("pos.dat")
dataset=pos.reshape(nf,N*3)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/pos.npy",data_tr,dtype='float32')
np.save("dataset/test/pos.npy",data_ts,dtype='float32')

box=np.loadtxt("box.dat")
dataset=box.reshape(nf,6)
np.random.seed(1)
np.random.shuffle(dataset)
data_tr,data_ts = train_test_split(dataset,train_size=0.8,random_state=seed_shuffle,shuffle=True)
np.save("dataset/training/box.npy",data_tr,dtype='float32')
np.save("dataset/test/box.npy",data_ts,dtype='float32')
