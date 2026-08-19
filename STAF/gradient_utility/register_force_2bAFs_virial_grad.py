import tensorflow as tf

from tensorflow.python.framework import ops

from staf_paths import code_root

def _ops(rel):
    return str(code_root() / rel)

compforcradgrad_module = tf.load_op_library(_ops('src/grad_force/rad/reforce.so'))


@ops.RegisterGradient("ComputeForceRadialVirial")
def _compute_force_radial_virial_grad(op, grad_force, grad_virial):
    if grad_force is None:
        grad_force = tf.zeros_like(op.outputs[0])
    if grad_virial is None:
        grad_virial = tf.zeros_like(op.outputs[1])

    [net_grad0, net_grad1, grad_2bemb_par] = (
        compforcradgrad_module.compute_force_radial_virial_grad(
            grad_force,
            grad_virial,
            op.inputs[0],
            op.inputs[1],
            op.inputs[2],
            op.inputs[3],
            op.inputs[4],
            op.inputs[5],
            op.inputs[6],
            op.inputs[7],
            op.inputs[8],
            op.inputs[9],
            op.inputs[10],
        )
    )
    # inputs: netderiv, desder, intmap, desr, alpha, emb, typemap, tipos, actual, pos, box
    return [net_grad0, None, None, None, net_grad1, grad_2bemb_par,
            None, None, None, None, None]
