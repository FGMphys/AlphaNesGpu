import tensorflow as tf

from staf_paths import code_root

def _ops(rel):
    return str(code_root() / rel)

class force_layer(tf.Module):
      def __init__(self,radbuff,angbuff):
          self.force2b=tf.load_op_library(_ops('src/force/rad/reforce.so'))
          self.force3b=tf.load_op_library(_ops('src/force/ang/reforce.so'))

          self.gradforce2b=tf.load_op_library(_ops('src/grad_force/rad/reforce.so'))
          self.gradforce3b=tf.load_op_library(_ops('src/grad_force/ang/reforce.so'))


          self.force2b.init_force_radial(radbuff)
          self.force3b.init_force_tripl(angbuff)

          self.gradforce2b.init_grad_force_radial(radbuff)
          self.gradforce3b.init_grad_force_tripl(angbuff)


      @tf.function()
      def __call__(self,net_der_r,x2b,intder2b,int2b,alpha2b,net_der_a,
                   x3b,x3bsupp,intder3b,intder3bsupp,int3b,numtriplet,alpha3b,
                   type_emb_2b,type_emb_3b,type_map,tipos,type_now):

          self.type_emb_2b_sq=tf.square(type_emb_2b)
          self.type_emb_3b_sq=tf.square(type_emb_3b)

          self.force_radial=self.force2b.compute_force_radial(net_der_r,intder2b,int2b,
                                                              x2b,alpha2b,self.type_emb_2b_sq,
                                                              type_map,tipos,type_now)
          self.force_angular=self.force3b.compute_force_tripl(net_der_a,x3bsupp,x3b,
                                                              intder3bsupp,intder3b,
                                                              int2b,int3b,alpha3b,
                                                              self.type_emb_3b_sq,
                                                              type_map,tipos,type_now,
                                                              numtriplet)
          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot



class force_debug_layer(tf.Module):
      def __init__(self,radbuff,angbuff):
          self.force2b=tf.load_op_library(_ops('src/force/rad/reforce.so'))
          self.force3b=tf.load_op_library(_ops('src/force/ang/reforce.so'))

          self.force2b.init_force_radial(radbuff)
          self.force3b.init_force_tripl(angbuff)

      @tf.function()
      def __call__(self,net_der_r,x2b,intder2b,int2b,alpha2b,net_der_a,
                   x3b,x3bsupp,intder3b,intder3bsupp,int3b,numtriplet,alpha3b,
                   type_emb_2b,type_emb_3b,type_map,tipos,type_now):

          self.type_emb_2b_sq=tf.square(type_emb_2b)
          self.type_emb_3b_sq=tf.square(type_emb_3b)

          self.force_radial=self.force2b.compute_force_radial(net_der_r,intder2b,int2b,
                                                              x2b,alpha2b,self.type_emb_2b_sq,
                                                              type_map,tipos,type_now)
          self.force_angular=self.force3b.compute_force_tripl(net_der_a,x3bsupp,x3b,
                                                              intder3bsupp,intder3b,
                                                              int2b,int3b,alpha3b,
                                                              self.type_emb_3b_sq,
                                                              type_map,tipos,type_now,
                                                              numtriplet)
          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot,self.force_radial,self.force_angular


class force_virial_layer(tf.Module):
      """Unified F + full virial tensor W (batch,9), trainable with RegisterGradient."""

      def __init__(self, radbuff, angbuff, with_grad=True):
          self.force2b = tf.load_op_library(_ops('src/force/rad/reforce.so'))
          self.force3b = tf.load_op_library(_ops('src/force/ang/reforce.so'))
          self.force2b.init_force_radial(radbuff)
          self.force3b.init_force_tripl(angbuff)
          if with_grad:
              self.gradforce2b = tf.load_op_library(_ops('src/grad_force/rad/reforce.so'))
              self.gradforce3b = tf.load_op_library(_ops('src/grad_force/ang/reforce.so'))
              self.gradforce2b.init_grad_force_radial(radbuff)
              self.gradforce3b.init_grad_force_tripl(angbuff)

      @tf.function()
      def __call__(self, net_der_r, x2b, intder2b, int2b, alpha2b, net_der_a,
                   x3b, x3bsupp, intder3b, intder3bsupp, int3b, numtriplet, alpha3b,
                   type_emb_2b, type_emb_3b, type_map, tipos, type_now, pos, box):
          type_emb_2b_sq = tf.square(type_emb_2b)
          type_emb_3b_sq = tf.square(type_emb_3b)

          force_radial, vir_rad = self.force2b.compute_force_radial_virial(
              net_der_r, intder2b, int2b, x2b, alpha2b, type_emb_2b_sq,
              type_map, tipos, type_now, pos, box)
          force_angular, vir_ang = self.force3b.compute_force_tripl_virial(
              net_der_a, x3bsupp, x3b, intder3bsupp, intder3b,
              int2b, int3b, alpha3b, type_emb_3b_sq,
              type_map, tipos, type_now, numtriplet, pos, box)
          return force_radial + force_angular, vir_rad + vir_ang
