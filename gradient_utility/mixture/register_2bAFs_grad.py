import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

root_path='/home/francegm/AlphaNesGpu'
compute_2b_pargrad = tf.load_op_library(root_path+'/src/mixture/grad_finger/rad/reforce.so')

@ops.RegisterGradient("ComputeSortProj")
def _compute_sort_proj_grad(op,grad):
    [net_grad,grad_emb2b_afs] =  compute_2b_pargrad.compute_two_body_par_grad (
                                                 grad,op.inputs[0],
                                                 op.inputs[1],op.inputs[2],
                                                 op.inputs[3],op.inputs[4]
                                                 )
    return [None,None,net_grad,grad_emb2b_afs,None]
