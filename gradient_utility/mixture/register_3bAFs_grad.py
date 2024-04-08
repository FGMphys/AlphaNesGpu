import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

root_path='/leonardo/pub/userexternal/fguidare/AlphaNesGpu'
module_alpha3body_grad = tf.load_op_library(root_path+'/src/mixture/grad_finger/ang/reforce.so')

@ops.RegisterGradient("ComputeSortProj3body")
def _compute_sort_proj_3body_grad(op, grad):
    compute_method =  module_alpha3body_grad.compute_sort_proj3body_grad

    [alpha3bodygrad,grad_emb3b_afs]=compute_method(grad,op.inputs[0],op.inputs[1],
                                  op.inputs[2],op.inputs[3],
                                  op.inputs[4],op.inputs[5],
                                  op.inputs[6],op.inputs[7])
    return [None, None, None, None,alpha3bodygrad,grad_emb3b_afs,None,None]
