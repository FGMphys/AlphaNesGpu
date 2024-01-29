import tensorflow as tf
import numpy as np
import sys
import time

def compare_routine(res,res_light):
    dim=len(res)
    for k in range(dim):
        print("Checking dataset ",k)
        control=np.sum(res[k].numpy()!=res_light[k].numpy())
        print(control)
        if control!=0:
           index=res[k].numpy()!=res_light[k].numpy()
           print(res[k].numpy()[index]-res_light[k].numpy()[index])


root_path='/home/francegm/Scrivania/Programmi/alpha_nes/alpha_nes_local_dev/descriptors_utility/distangoli_py'
descriptor_sopath=root_path+'/op_comp_3bcc.so'


descriptor_light_sopath=root_path+'/op_comp_3bcc_light.so'

make_descriptor=tf.load_op_library(descriptor_sopath)
make_descriptor_light=tf.load_op_library(descriptor_light_sopath)

pos=tf.constant(np.loadtxt(sys.argv[1]).reshape(-1,1),dtype='float64')
box=tf.constant(np.loadtxt(sys.argv[2],dtype='float64'))


#Input("radial_cutoff: double")
#    .Input("radial_buffer: int32")
#    .Input("angular_cutoff: double")
#    .Input("angular_buffer: int32")
#    .Input("numpar: int32")
#    .Input("positions: double")
#    .Input("box_dim: int32")
#    .Input("boxer: double")
#    .Input("number_of_frames: int32")
#    .Output("descriptors: double")
#    .Output("des3bsupp: double")
#    .Output("intmap2b: int32")
#    .Output("intmap3b: int32")
#    .Output("der2b: double")
#    .Output("der3b: double")
#    .Output("der3bsupp: double");

N=1000
nf=pos.shape[1]
print("Ecooo",nf)
#[des,des3bsupp,intmap2b,intmap3b,der2b,der3b,der3bsupp]
start=time.time()
res=make_descriptor.compute_descriptors(1.8,30,1.8,300,1000,pos,6,box,nf)
stop=time.time()
print("Computed first routine with ",stop-start," s")
print(res[0][0,0])
#[des_light,des3bsupp_light,intmap2b_light,intmap3b_light,der2b_light,der3b_light,der3bsupp_light]=
start=time.time()
#   .Input("radial_cutoff: double")
#   .Input("radial_buffer: int32")
#   .Input("angular_buffer: int32")
#   .Input("numpar: int32")
#   .Input("boxer: double");
make_descriptor_light.construct_descriptors_light(1.81,30,300,1000,box)
print("Computed the construction!")
res_light=make_descriptor_light.compute_descriptors_light(1.8,30,1.8,300,1000,pos,box,nf)
print(res_light[0][0,0])
stop=time.time()
print("Computed second routine with ",stop-start," s")
#res=[des,des3bsupp,intmap2b,intmap3b,der2b,der3b,der3bsupp]
compare_routine(res,res_light)
label=['des','des3bsupp','intmap2b','intmap3b','der2b','der3b','der3bsupp']
for k,el in enumerate(res):
    if (label[k]!='der2b' and label[k]!='der3b') and label[k]!='der3bsupp':
       np.savetxt(label[k]+"py_routine",el.numpy().reshape(N,-1),fmt='%14.11g',delimiter=',')
    else:
       np.savetxt(label[k]+"py_routine",el.numpy().reshape(3*N,-1),fmt='%14.11g',delimiter=',')
breakpoint()
