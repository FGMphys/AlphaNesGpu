import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from staf.dtype import tf_dtype
import pickle


class alpha_nes_full(tf.Module):
    def __init__(self,physics_layer,force_layer,output_dim,lossfunction,val_loss,
                opt_net,alpha_bound,lognorm_layer,color_type_map,
                restart,seed_fix,full_param):
        super(alpha_nes_full, self).__init__()

        tf.keras.utils.set_random_seed(seed_fix)
        #self.tipos=tf.constant(tipos,dtype='int32')
        map_NN_layer=full_param['map_NN_layer']
        number_of_NN=len(map_NN_layer)
        node_list=[map_NN_layer[key] for key in map_NN_layer.keys()]

        map_actfun=full_param['activation_function']
        self.number_of_NN=number_of_NN
        self.color_type_map=color_type_map
        self.map_color_interaction=np.loadtxt(full_param['color_interaction_file'],dtype='int32').reshape((-1,1))
        self.map_intra=np.loadtxt(full_param['map_intra_file'],dtype='int32').reshape((-1,1)).reshape((-1,1))

        self.N=len(color_type_map)

        #Atomic Finger Print Layer
        self.physics_layer=[physlay_type for physlay_type in physics_layer]
        self.lognorm_layer=[lognorlay_type for lognorlay_type in lognorm_layer]
        #Force layer for force calculation
        self.force_layer=force_layer

        #Dense Net layers
        if restart=='no' or restart=='only_afs':

            self.nets = [tf.keras.Sequential() for el in range(self.number_of_NN)]
            for index_net,net in enumerate(self.nets):
                net.add(Input(shape=(self.N,self.physics_layer[index_net].output_dim,)))
                if len(map_NN_layer[index_net])>0:
                    for k in node_list[index_net]:
                        net.add(Dense(k, activation=map_actfun[index_net]))
                    net.add(Dense(output_dim))
        else:
             print("alpha_nes: parameters of NN loaded from ",restart)
             self.nets=[tf.keras.models.load_model(restart+'/net_model_type'+str(k))
                       for k in range(self.number_of_NN)]
             if full_param['restart']!='all_params':
                with open(restart+'/opt_net_weights','rb') as source:
                     weight_net=pickle.load(source)
                self.opt_net_weights=weight_net
                print("alpha_nes: State of optimizer has been loaded from ",restart)
                #res=[self.nets[k].compile() for k in range(self.ntipos)]
        self.lossfunction=lossfunction
        self.val_loss=val_loss

        self.relu_bound=tf.keras.layers.ReLU(max_value=None,
                                           threshold=alpha_bound,
                                           negative_slope=0.0)

        self.opt_net=opt_net
        self.global_step=0


    @tf.function()
    def full_train_e(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue,pe,pf,pb):

        self.x2b = x1
        self.x3b = x2
        self.x3bsupp = x3bsupp
        self.int2b = int2b
        self.int3b = int3b
        self.numtriplet = numtriplet


        number_of_NN=self.number_of_NN
        self.fingerprint=[self.physics_layer[k](self.x2b,self.x3bsupp,
        self.int2b,self.x3b,self.int3b,self.numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]


        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5


        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_bound_2b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha2b))
              for k in range(number_of_NN)]
        loss_bound_3b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha3b))
                 for k in range(number_of_NN)]
        loss_bound=tf.add_n(loss_bound_2b)+tf.add_n(loss_bound_3b)

        loss=pe*loss_energy+pb*loss_bound



        grad_w=[tf.gradients(loss,net.trainable_variables) for net  in  self.nets]
        grad_2b=[tf.gradients(loss,physlay.alpha2b) for physlay in self.physics_layer]
        grad_3b=[tf.gradients(loss,physlay.alpha3b) for physlay in self.physics_layer]
        grad_mu=[tf.gradients(loss,lognorm.mu) for lognorm in self.lognorm_layer]

        ########MODIFICA 19.04: SOLVE BUG ONLY FIRST HIDDEN ###########
        #all_net_grad=[grad_w[k] for k in range(nt)]
        #all_net_param=[self.nets[k].trainable_variables for k in range(nt)]
        grads_and_vars_net=[]
        for k in range(number_of_NN):
            grad_var_pairs = [(grad, param) for grad, param in zip(grad_w[k], self.nets[k].trainable_variables)]
            grads_and_vars_net.extend(grad_var_pairs)
        grads_and_vars_afs = []

        for k in range(number_of_NN):
            # Aggiungi i gradienti per alpha2b
            grads_and_vars_afs.extend([(grad_2b[k][0], self.physics_layer[k].alpha2b)])

            # Aggiungi i gradienti per alpha3b
            grads_and_vars_afs.extend([(grad_3b[k][0], self.physics_layer[k].alpha3b)])

            # Aggiungi i gradienti per mu
            grads_and_vars_afs.extend([(grad_mu[k][0], self.lognorm_layer[k].mu)])

        grads_and_vars_all = grads_and_vars_afs + grads_and_vars_net

        self.opt_net.apply_gradients(grads_and_vars_all)

        self.global_step=self.global_step+1
        ######## FINE MODIFICA 19.04: SOLVE BUG ONLY FIRST HIDDEN ###########
        return loss_force+loss_energy,loss_force,loss_energy

    @tf.function()
    def full_test_e(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,
                    numtriplet,etrue,ftrue):


        self.x2b = x1
        self.x3b = x2
        self.x3bsupp = x3bsupp
        self.int2b = int2b
        self.int3b = int3b
        self.numtriplet = numtriplet


        number_of_NN=self.number_of_NN
        self.fingerprint=[self.physics_layer[k](self.x2b,self.x3bsupp,
        self.int2b,self.x3b,self.int3b,self.numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]


        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5


        loss_energy=self.val_loss(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype=tf_dtype())
        loss=loss_energy+loss_force

        return loss, loss_force,loss_energy
    @tf.function()
    def full_train_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue,pe,pf,pb):

        self.x2b = x1
        self.x3b = x2
        self.x3bsupp = x3bsupp
        self.int2b = int2b
        self.int3b = int3b
        self.intder2b = intder2b
        self.intder3b = intder3b
        self.intder3bsupp = intder3bsupp
        self.numtriplet = numtriplet

        number_of_NN=self.number_of_NN
        self.fingerprint=[self.physics_layer[k](self.x2b,self.x3bsupp,
        self.int2b,self.x3b,self.int3b,self.numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]


        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5

        self.grad_listed=[tf.split(self.grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(number_of_NN)]

        self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b,
                                 self.intder2b,self.int2b,
                                 self.physics_layer[k].alpha2b,
                                 self.grad_listed[k][1],self.x3b,self.x3bsupp,
                                 self.intder3b,self.intder3bsupp,self.int3b,
                                 self.numtriplet,
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.color_type_map,self.map_color_interaction,
                                 k,self.map_intra) for k in range(number_of_NN)]

        self.force=tf.math.add_n(self.force_list)

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=self.lossfunction(self.force,ftrue)
        loss_bound_2b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha2b))
              for k in range(number_of_NN)]
        loss_bound_3b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha3b))
                 for k in range(number_of_NN)]
        loss_bound=tf.add_n(loss_bound_2b)+tf.add_n(loss_bound_3b)

        loss=pb*loss_bound+pf*loss_force#pe*loss_energy+l1_loss



        grad_w=[tf.gradients(loss,net.trainable_variables) for net  in  self.nets]
        grad_2b=[tf.gradients(loss,physlay.alpha2b) for physlay in self.physics_layer]
        grad_3b=[tf.gradients(loss,physlay.alpha3b) for physlay in self.physics_layer]
        grad_mu=[tf.gradients(loss,lognorm.mu) for lognorm in self.lognorm_layer]

        ########MODIFICA 19.04: SOLVE BUG ONLY FIRST HIDDEN ###########
        #all_net_grad=[grad_w[k] for k in range(nt)]
        #all_net_param=[self.nets[k].trainable_variables for k in range(nt)]
        grads_and_vars_net=[]
        for k in range(number_of_NN):
            grad_var_pairs = [(grad, param) for grad, param in zip(grad_w[k], self.nets[k].trainable_variables)]
            grads_and_vars_net.extend(grad_var_pairs)
        grads_and_vars_afs = []

        for k in range(number_of_NN):
            # Aggiungi i gradienti per alpha2b
            grads_and_vars_afs.extend([(grad_2b[k][0], self.physics_layer[k].alpha2b)])

            # Aggiungi i gradienti per alpha3b
            grads_and_vars_afs.extend([(grad_3b[k][0], self.physics_layer[k].alpha3b)])

            # Aggiungi i gradienti per mu
            grads_and_vars_afs.extend([(grad_mu[k][0], self.lognorm_layer[k].mu)])

        grads_and_vars_all = grads_and_vars_afs + grads_and_vars_net
        self.opt_net.apply_gradients(grads_and_vars_all)
        self.global_step=self.global_step+1
        ######## FINE MODIFICA 19.04: SOLVE BUG ONLY FIRST HIDDEN ###########

        return loss,loss_energy,loss_bound,loss_force

    @tf.function()
    def full_loss_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue,pe,pf,pb):
        """Same Loss as full_train_e_f, without apply_gradients (FD / analytic grads)."""
        number_of_NN=self.number_of_NN
        fingerprint=[self.physics_layer[k](x1,x3bsupp,
        int2b,x2,int3b,numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]
        log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(fingerprint)]
        energy=[self.nets[k](cp) for k,cp in enumerate(log_norm_projdes)]
        grad_ene=[tf.gradients(energy[k],cp) for k,cp in enumerate(fingerprint)]
        totene=tf.concat(energy,axis=1)
        totenergy=tf.reduce_mean(totene,axis=(-1,-2))*0.5
        grad_listed=[tf.split(grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(number_of_NN)]
        force_list=[self.force_layer(grad_listed[k][0],x1,
                                 intder2b,int2b,
                                 self.physics_layer[k].alpha2b,
                                 grad_listed[k][1],x2,x3bsupp,
                                 intder3b,intder3bsupp,int3b,
                                 numtriplet,
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.color_type_map,self.map_color_interaction,
                                 k,self.map_intra) for k in range(number_of_NN)]
        force=tf.math.add_n(force_list)
        loss_energy=self.lossfunction(totenergy,etrue)
        loss_force=self.lossfunction(force,ftrue)
        loss_bound_2b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha2b))
              for k in range(number_of_NN)]
        loss_bound_3b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha3b))
                 for k in range(number_of_NN)]
        loss_bound=tf.add_n(loss_bound_2b)+tf.add_n(loss_bound_3b)
        loss=pb*loss_bound+pf*loss_force
        grad_w=[tf.gradients(loss,net.trainable_variables) for net in self.nets]
        grad_2b=[tf.gradients(loss,physlay.alpha2b) for physlay in self.physics_layer]
        grad_3b=[tf.gradients(loss,physlay.alpha3b) for physlay in self.physics_layer]
        return loss,loss_energy,loss_bound,loss_force,grad_w,grad_2b,grad_3b

    @tf.function()
    def full_mse_grads_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue):
        """MSE Loss_E / Loss_F + analytic param grads (no bound, no apply_gradients)."""
        number_of_NN=self.number_of_NN
        fingerprint=[self.physics_layer[k](x1,x3bsupp,
        int2b,x2,int3b,numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]
        log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(fingerprint)]
        energy=[self.nets[k](cp) for k,cp in enumerate(log_norm_projdes)]
        grad_ene=[tf.gradients(energy[k],cp) for k,cp in enumerate(fingerprint)]
        totene=tf.concat(energy,axis=1)
        totenergy=tf.reduce_mean(totene,axis=(-1,-2))*0.5
        grad_listed=[tf.split(grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(number_of_NN)]
        force_list=[self.force_layer(grad_listed[k][0],x1,
                                 intder2b,int2b,
                                 self.physics_layer[k].alpha2b,
                                 grad_listed[k][1],x2,x3bsupp,
                                 intder3b,intder3bsupp,int3b,
                                 numtriplet,
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.color_type_map,self.map_color_interaction,
                                 k,self.map_intra) for k in range(number_of_NN)]
        force=tf.math.add_n(force_list)
        loss_energy=self.val_loss(totenergy,etrue)
        loss_force=self.val_loss(force,ftrue)
        vars_net=self.nets[0].trainable_variables
        probe_vars=[vars_net[0], vars_net[1], self.physics_layer[0].alpha2b, self.physics_layer[0].alpha3b]
        grad_E=[tf.gradients(loss_energy,v)[0] for v in probe_vars]
        grad_F=[tf.gradients(loss_force,v)[0] for v in probe_vars]
        return loss_energy,loss_force,grad_E,grad_F

    @tf.function()
    def full_test_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,
                     numtriplet,etrue,ftrue):
        self.x2b = x1
        self.x3b = x2
        self.x3bsupp = x3bsupp
        self.int2b = int2b
        self.int3b = int3b
        self.intder2b = intder2b
        self.intder3b = intder3b
        self.intder3bsupp = intder3bsupp
        self.numtriplet = numtriplet

        number_of_NN=self.number_of_NN
        self.fingerprint=[self.physics_layer[k](self.x2b,self.x3bsupp,
        self.int2b,self.x3b,self.int3b,self.numtriplet,self.color_type_map,
        self.map_color_interaction,self.map_intra)
                    for k in range(number_of_NN)]

        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]

        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]

        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5
        self.grad_listed=[tf.split(self.grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(number_of_NN)]
        self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b,
                                 self.intder2b,self.int2b,
                                 self.physics_layer[k].alpha2b,
                                 self.grad_listed[k][1],self.x3b,self.x3bsupp,
                                 self.intder3b,self.intder3bsupp,self.int3b,
                                 self.numtriplet,
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.color_type_map,self.map_color_interaction,k,
                                 self.map_intra) for k in range(number_of_NN)]

        self.force=tf.math.add_n(self.force_list)



        loss_energy=self.val_loss(self.totenergy,etrue)
        loss_force=self.val_loss(self.force,ftrue)
        loss=loss_energy+loss_force
        return loss, loss_force,loss_energy


    def save_model(self,folder_ou):
        for k,net in enumerate(self.nets):
            net.save(folder_ou+'/net_model_type'+str(k),overwrite=True)
            np.savetxt(folder_ou+'/type'+str(k)+'_alpha_2body.dat',self.physics_layer[k].alpha2b.numpy())
            np.savetxt(folder_ou+'/type'+str(k)+'_alpha_3body.dat',self.physics_layer[k].alpha3b.numpy())
            np.savetxt(folder_ou+'/type'+str(k)+'_type_emb_2b.dat',self.physics_layer[k].type_emb_2b.numpy())
            np.savetxt(folder_ou+'/type'+str(k)+'_type_emb_3b.dat',self.physics_layer[k].type_emb_3b.numpy())
            np.savetxt(folder_ou+'/type'+str(k)+'_type_emb_2b_sq.dat',self.physics_layer[k].type_emb_2b.numpy()**2)
            np.savetxt(folder_ou+'/type'+str(k)+'_type_emb_3b_sq.dat',self.physics_layer[k].type_emb_3b.numpy()**2)
            np.savetxt(folder_ou+'/type'+str(k)+'_alpha_mu.dat',self.lognorm_layer[k].mu.numpy())
            np.savetxt(folder_ou+'/color_type_map.dat',self.color_type_map)
            np.savetxt(folder_ou+'/map_color_interaction.dat',self.map_color_interaction)
            np.savetxt(folder_ou+'/map_intra.dat',self.map_intra)
            with open(folder_ou+'/opt_net_weights','wb') as dest:
                 pickle.dump(self.opt_net.variables(),dest)
            with open(folder_ou+'/opt_net_conf','wb') as dest:
                 pickle.dump(self.opt_net.get_config(),dest)
    def save_model_init(self,folder_ou):
        for k,net in enumerate(self.nets):
            net.save(folder_ou+'/init_net_model_type'+str(k),overwrite=True)

    def build_opt_weights(self):
        self.opt_net.build(self.opt_net_weights)
    def set_opt_weight(self):
        self.opt_net.set_weights(self.opt_net_weights)
    def get_op_weigth(self):
        return self.opt_net.variables()
    def get_lrnet(self):
        return self.opt_net.learning_rate
