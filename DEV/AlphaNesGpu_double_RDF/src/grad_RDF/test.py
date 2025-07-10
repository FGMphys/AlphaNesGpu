import tensorflow as tf
import numpy as np

RDFop=tf.load_op_library("reforce.so")


box=np.loadtxt("/home/francegm/NN_HS_Frank/model0/scratch/box_dataset.dat")
nf=np.shape(box)[0]
position=np.loadtxt("/home/francegm/NN_HS_Frank/model0/scratch/pos_dataset.dat").reshape((nf,-1))
ground_energy=np.loadtxt("/home/francegm/NN_HS_Frank/model0/energy.dat")
betarewe=1.
binsize=0.01
newenergy=np.loadtxt("/home/francegm/NN_HS_Frank/model0/energy.dat")
type_map=np.loadtxt("/home/francegm/NN_HS_Frank/model0/type_map.dat").astype("int")
type_vec=np.array([1000]).astype("int")
type_pair=np.array([0,0])

breakpoint()

new_RDF=RDFop.compute_rdf(newenergy,position,box,ground_energy,type_pair,type_vec,type_map,binsize,betarewe);

breakpoint()
