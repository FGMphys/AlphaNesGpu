import tensorflow as tf

from tensorflow.python.framework import ops

from staf_cg_paths import code_root

def _ops(rel):
    return str(code_root() / rel)

compforcegradtripl_module = tf.load_op_library(_ops('src/grad_force/ang/reforce.so'))


@ops.RegisterGradient("ComputeForceTriplVirial")
def _compute_force_tripl_virial_grad(op, grad_force, grad_virial):
    if grad_force is None:
        grad_force = tf.zeros_like(op.outputs[0])
    if grad_virial is None:
        grad_virial = tf.zeros_like(op.outputs[1])

    [net_grad0, net_grad1, grad_emb3b_par] = (
        compforcegradtripl_module.compute_force_tripl_virial_grad(
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
            op.inputs[11],
            op.inputs[12],
            op.inputs[13],
            op.inputs[14],
            op.inputs[15],
        )
    )
    # 16 CG inputs: netderiv, desr, desa, intder_r, intder_a, intmap_r, intmap_a,
    #               alpha3b, emb3b, color_type_map, map_color_interaction,
    #               actual, numtriplet, map_intra, pos, box
    # grads on netderiv, alpha3b, emb3b
    return [net_grad0, None, None, None, None, None, None, net_grad1, grad_emb3b_par,
            None, None, None, None, None, None, None]
