import tensorflow as tf

from staf_cg_paths import code_root

def _ops(rel):
    return str(code_root() / rel)

force2b_sopath=_ops('src/force/rad/reforce.so')
force3b_sopath=_ops('src/force/ang/reforce.so')

gradforce2b_sopath=_ops('src/grad_force/rad/reforce.so')
gradforce3b_sopath=_ops('src/grad_force/ang/reforce.so')

class force_layer(tf.Module):
      def __init__(self,radbuff,angbuff):
          self.force2b=tf.load_op_library(force2b_sopath)
          self.force3b=tf.load_op_library(force3b_sopath)

          self.gradforce2b=tf.load_op_library(gradforce2b_sopath)
          self.gradforce3b=tf.load_op_library(gradforce3b_sopath)


          self.force2b.init_force_radial(radbuff)
          self.force3b.init_force_tripl(angbuff)

          self.gradforce2b.init_grad_force_radial(radbuff)
          self.gradforce3b.init_grad_force_tripl(angbuff)


      @tf.function()
      def __call__(self,net_der_r,x2b,intder2b,int2b,alpha2b,net_der_a,
                   x3b,x3bsupp,intder3b,intder3bsupp,int3b,numtriplet,alpha3b,
                   type_emb_2b,type_emb_3b,color_type_map,map_color_interaction,type_now,
                   map_intra):

          self.type_emb_2b_sq=tf.square(type_emb_2b)
          self.type_emb_3b_sq=tf.square(type_emb_3b)
          #qui camilla mi distrae se c'e' bug colpa sua
          self.force_radial=self.force2b.compute_force_radial(net_der_r,intder2b,int2b,
                                                              x2b,alpha2b,self.type_emb_2b_sq,
                                                              color_type_map,map_color_interaction,
                                                              type_now,map_intra)
          self.force_angular=self.force3b.compute_force_tripl(net_der_a,x3bsupp,x3b,
                                                              intder3bsupp,intder3b,
                                                              int2b,int3b,alpha3b,
                                                              self.type_emb_3b_sq,
                                                              color_type_map,map_color_interaction,
                                                              type_now,numtriplet,map_intra)
          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot



class force_debug_layer(tf.Module):
      def __init__(self,radbuff,angbuff):
          self.force2b=tf.load_op_library(force2b_sopath)
          self.force3b=tf.load_op_library(force3b_sopath)

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
                                                              color_type_map,map_color_interaction,
                                                              type_now,map_intra)
          self.force_angular=self.force3b.compute_force_tripl(net_der_a,x3bsupp,x3b,
                                                              intder3bsupp,intder3b,
                                                              int2b,int3b,alpha3b,
                                                              self.type_emb_3b_sq,
                                                              color_type_map,map_color_interaction,
                                                              type_now,numtriplet,map_intra)

          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot,self.force_radial,self.force_angular


class force_virial_layer(tf.Module):
      """Unified F + full virial tensor W (batch,9), trainable with RegisterGradient."""

      def __init__(self, radbuff, angbuff, with_grad=True):
          self.force2b = tf.load_op_library(force2b_sopath)
          self.force3b = tf.load_op_library(force3b_sopath)
          self.force2b.init_force_radial(radbuff)
          self.force3b.init_force_tripl(angbuff)
          if with_grad:
              self.gradforce2b = tf.load_op_library(gradforce2b_sopath)
              self.gradforce3b = tf.load_op_library(gradforce3b_sopath)
              self.gradforce2b.init_grad_force_radial(radbuff)
              self.gradforce3b.init_grad_force_tripl(angbuff)

      @tf.function()
      def __call__(self, net_der_r, x2b, intder2b, int2b, alpha2b, net_der_a,
                   x3b, x3bsupp, intder3b, intder3bsupp, int3b, numtriplet, alpha3b,
                   type_emb_2b, type_emb_3b, color_type_map, map_color_interaction,
                   type_now, map_intra, pos, box):
          type_emb_2b_sq = tf.square(type_emb_2b)
          type_emb_3b_sq = tf.square(type_emb_3b)

          force_radial, vir_rad = self.force2b.compute_force_radial_virial(
              net_der_r, intder2b, int2b, x2b, alpha2b, type_emb_2b_sq,
              color_type_map, map_color_interaction, type_now, map_intra, pos, box)
          force_angular, vir_ang = self.force3b.compute_force_tripl_virial(
              net_der_a, x3bsupp, x3b, intder3bsupp, intder3b,
              int2b, int3b, alpha3b, type_emb_3b_sq,
              color_type_map, map_color_interaction, type_now, numtriplet,
              map_intra, pos, box)
          return force_radial + force_angular, vir_rad + vir_ang
