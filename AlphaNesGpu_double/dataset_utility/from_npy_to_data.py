import numpy as np

pos=np.load("pos.npy")
box=np.load("box.npy")
N=int(pos.shape[1]/3)

for k,el in enumerate(pos):
    header_now=str(k)+" "+str(N)+" "+str(box[k,0])+" "+str(box[k,1])+" "+str(box[k,2])+" "+str(box[k,3])+" "+str(box[k,4])+" "+str(box[k,5])
    np.savetxt("pos_"+str(k),el.reshape((N,3)),header=header_now,comments='')


