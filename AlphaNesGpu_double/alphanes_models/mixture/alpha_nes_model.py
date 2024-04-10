import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
import pickle


class alpha_nes_full(tf.Module):
    def __init__(self,physics_layer,force_layer,num_layers,node_seq,actfun,
               output_dim,lossfunction,val_loss,opt_net,opt_phys,alpha_bound,
               lognorm_layer,tipos,type_map,restart,seed_fix):
        super(alpha_nes_full, self).__init__()

        tf.keras.utils.set_random_seed(seed_fix)
        self.tipos=tf.constant(tipos,dtype='int32')
        self.ntipos=len(tipos)
        self.type_map=type_map

        self.N=len(type_map)

        self.physics_layer=[physlay_type for physlay_type in physics_layer]
        self.lognorm_layer=[lognorlay_type for lognorlay_type in lognorm_layer]
        self.force_layer=force_layer


        if restart=='no' or restart=='only_afs':
            self.nhl=num_layers
            self.node=node_seq
            self.actfun=actfun

            self.nets = [tf.keras.Sequential() for el in range(self.ntipos)]
            for ntype,net in enumerate(self.nets):
                net.add(Input(shape=(tipos[ntype],self.physics_layer[ntype].output_dim,)))
                if self.nhl>0:
                    for k in self.node:
                        net.add(Dense(k, activation=self.actfun))
                    net.add(Dense(output_dim))
        else:
             self.nets=[tf.keras.models.load_model(restart+'/net_model_type'+str(k))
                       for k in range(self.ntipos)]
             if restart!='all_params':
                with open(restart+'/opt_net_weights','rb') as source:
                     weight_net=pickle.load(source)
                self.opt_net_weights=weight_net
                with open(restart+'/opt_phys_weights','rb') as source:
                     weight_phys=pickle.load(source)
                self.opt_phys_weights=weight_phys
                #res=[self.nets[k].compile() for k in range(self.ntipos)]
        self.lossfunction=lossfunction
        self.val_loss=val_loss

        self.relu_bound=tf.keras.layers.ReLU(max_value=None,
                                           threshold=alpha_bound,
                                           negative_slope=0.0)

        self.opt_net=opt_net
        self.opt_phys=opt_phys
        self.global_step=0


    @tf.function()
    def full_train_e(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue,pe,pf,pb):

        nt=self.ntipos
        self.x2b=tf.split(x1,self.tipos,axis=1)
        self.x3b=tf.split(x2,self.tipos,axis=1)
        self.x3bsupp=tf.split(x3bsupp,self.tipos,axis=1)
        self.int2b=tf.split(int2b,self.tipos,axis=1)
        self.int3b=tf.split(int3b,self.tipos,axis=1)
        self.numtriplet=tf.split(numtriplet,self.tipos,axis=1)

        self.fingerprint=[self.physics_layer[k](self.x2b[k],self.x3bsupp[k],
        self.int2b[k],self.x3b[k],self.int3b[k],self.numtriplet[k],
        self.type_map) for k in range(nt)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]

        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5


        loss_bound_2b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha2b))
              for k in range(nt)]
        loss_bound_3b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha3b))
                 for k in range(nt)]
        loss_bound=tf.add_n(loss_bound_2b)+tf.add_n(loss_bound_3b)

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype='float64')
        loss=pe*loss_energy+pb*loss_bound
        grad_w=[tf.gradients(loss,net.trainable_variables) for net  in  self.nets]
        grad_2b=[tf.gradients(loss,physlay.alpha2b) for physlay in self.physics_layer]
        grad_3b=[tf.gradients(loss,physlay.alpha3b) for physlay in self.physics_layer]
        grad_mu=[tf.gradients(loss,lognorm.mu) for lognorm in self.lognorm_layer]
        all_net_grad=[grad_w[k][0] for k in range(nt)]
        all_net_param=[self.nets[k].trainable_variables[0] for k in range(nt)]
        self.opt_net.apply_gradients(zip(all_net_grad,all_net_param))

        all_AFs_param=[self.physics_layer[k].alpha2b for k in range(nt)]
        for k in range(nt):
            all_AFs_param.append(self.physics_layer[k].alpha3b)
        for k in range(nt):
            all_AFs_param.append(self.lognorm_layer[k].mu)

        all_AFS_grad=[grad_2b[k][0] for k in range(nt)]
        for k in range(nt):
            all_AFS_grad.append(grad_3b[k][0])
        for k in range(nt):
            all_AFS_grad.append(grad_mu[k][0])
        self.opt_phys.apply_gradients(zip(all_AFS_grad,
                                              all_AFs_param))
        self.global_step=self.global_step+1
        return loss_force+loss_energy,loss_force,loss_energy

    @tf.function()
    def full_test_e(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,
                    numtriplet,etrue,ftrue):

        nt=self.ntipos



        self.x2b=tf.split(x1,self.tipos,axis=1)
        self.x3b=tf.split(x2,self.tipos,axis=1)
        self.x3bsupp=tf.split(x3bsupp,self.tipos,axis=1)
        self.int2b=tf.split(int2b,self.tipos,axis=1)
        self.int3b=tf.split(int3b,self.tipos,axis=1)
        self.numtriplet=tf.split(numtriplet,self.tipos,axis=1)


        self.fingerprint=[self.physics_layer[k](self.x2b[k],self.x3bsupp[k],
        self.int2b[k],self.x3b[k],self.int3b[k],self.numtriplet[k],self.type_map)
                        for k in range(nt)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                             for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]

        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5


        loss_energy=self.val_loss(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype='float64')
        loss=loss_energy+loss_force

        return loss, loss_force,loss_energy
    @tf.function()
    def full_train_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,numtriplet,etrue,ftrue,pe,pf,pb):

        nt=self.ntipos


        self.x2b=tf.split(x1,self.tipos,axis=1)
        self.x3b=tf.split(x2,self.tipos,axis=1)
        self.x3bsupp=tf.split(x3bsupp,self.tipos,axis=1)
        self.int2b=tf.split(int2b,self.tipos,axis=1)
        self.int3b=tf.split(int3b,self.tipos,axis=1)
        self.intder2b=tf.split(intder2b,self.tipos,axis=1)
        self.intder3b=tf.split(intder3b,self.tipos,axis=1)
        self.intder3bsupp=tf.split(intder3bsupp,self.tipos,axis=1)
        self.numtriplet=tf.split(numtriplet,self.tipos,axis=1)


        self.fingerprint=[self.physics_layer[k](self.x2b[k],self.x3bsupp[k],
        self.int2b[k],self.x3b[k],self.int3b[k],self.numtriplet[k],self.type_map)
                    for k in range(nt)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]
        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]


        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5

        self.grad_listed=[tf.split(self.grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(nt)]

        self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b[k],
                                 self.intder2b[k],self.int2b[k],
                                 self.physics_layer[k].alpha2b,
                                 self.grad_listed[k][1],self.x3b[k],self.x3bsupp[k],
                                 self.intder3b[k],self.intder3bsupp[k],self.int3b[k],
                                 self.numtriplet[k],
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.type_map,self.tipos,k) for k in range(nt)]

        self.force=tf.math.add_n(self.force_list)

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=self.lossfunction(self.force,ftrue)
        loss_bound_2b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha2b))
              for k in range(nt)]
        loss_bound_3b=[tf.math.reduce_sum(self.relu_bound(self.physics_layer[k].alpha3b))
                 for k in range(nt)]
        loss_bound=tf.add_n(loss_bound_2b)+tf.add_n(loss_bound_3b)

        loss=pe*loss_energy+pb*loss_bound+pf*loss_force#+l1_loss



        grad_w=[tf.gradients(loss,net.trainable_variables) for net  in  self.nets]
        grad_2b=[tf.gradients(loss,physlay.alpha2b) for physlay in self.physics_layer]
        grad_3b=[tf.gradients(loss,physlay.alpha3b) for physlay in self.physics_layer]
        grad_mu=[tf.gradients(loss,lognorm.mu) for lognorm in self.lognorm_layer]

        all_net_grad=[grad_w[k][0] for k in range(nt)]
        all_net_param=[self.nets[k].trainable_variables[0] for k in range(nt)]
        self.opt_net.apply_gradients(zip(all_net_grad,all_net_param))


        all_AFs_param=[self.physics_layer[k].alpha2b for k in range(nt)]
        for k in range(nt):
            all_AFs_param.append(self.physics_layer[k].alpha3b)
        for k in range(nt):
            all_AFs_param.append(self.lognorm_layer[k].mu)

        all_AFS_grad=[grad_2b[k][0] for k in range(nt)]
        for k in range(nt):
            all_AFS_grad.append(grad_3b[k][0])
        for k in range(nt):
            all_AFS_grad.append(grad_mu[k][0])
        self.opt_phys.apply_gradients(zip(all_AFS_grad,
                                              all_AFs_param))
        self.global_step=self.global_step+1
        return loss,loss_energy,loss_bound,loss_force

    @tf.function()
    def full_test_e_f(self,x1,x2,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,
                     numtriplet,etrue,ftrue):

        nt=self.ntipos



        self.x2b=tf.split(x1,self.tipos,axis=1)
        self.x3b=tf.split(x2,self.tipos,axis=1)
        self.x3bsupp=tf.split(x3bsupp,self.tipos,axis=1)
        self.int2b=tf.split(int2b,self.tipos,axis=1)
        self.int3b=tf.split(int3b,self.tipos,axis=1)
        self.intder2b=tf.split(intder2b,self.tipos,axis=1)
        self.intder3b=tf.split(intder3b,self.tipos,axis=1)
        self.intder3bsupp=tf.split(intder3bsupp,self.tipos,axis=1)
        self.numtriplet=tf.split(numtriplet,self.tipos,axis=1)


        self.fingerprint=[self.physics_layer[k](self.x2b[k],self.x3bsupp[k],
        self.int2b[k],self.x3b[k],self.int3b[k],self.numtriplet[k],self.type_map)
                    for k in range(nt)]
        self.log_norm_projdes=[self.lognorm_layer[k](finger)
                         for k,finger in enumerate(self.fingerprint)]

        self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
        self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]

        self.totene=tf.concat(self.energy,axis=1)
        self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5
        self.grad_listed=[tf.split(self.grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                    self.physics_layer[k].nalpha_a],axis=2) for k in range(nt)]

        self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b[k],
                                 self.intder2b[k],self.int2b[k],
                                 self.physics_layer[k].alpha2b,
                                 self.grad_listed[k][1],self.x3b[k],self.x3bsupp[k],
                                 self.intder3b[k],self.intder3bsupp[k],self.int3b[k],
                                 self.numtriplet[k],
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.type_map,self.tipos,k) for k in range(nt)]

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
            with open(folder_ou+'/opt_net_weights','wb') as dest:
                 pickle.dump(self.opt_net.variables(),dest)
            with open(folder_ou+'/opt_phys_weights','wb') as dest:
                 pickle.dump(self.opt_phys.variables(),dest)
            with open(folder_ou+'/opt_net_conf','wb') as dest:
                 pickle.dump(self.opt_net.get_config(),dest)
            with open(folder_ou+'/opt_phys_conf','wb') as dest:
                 pickle.dump(self.opt_phys.get_config(),dest)
    def save_model_init(self,folder_ou):
        for k,net in enumerate(self.nets):
            net.save(folder_ou+'/init_net_model_type'+str(k),overwrite=True)

    def set_opt_weight(self):
        self.opt_phys.set_weights(self.opt_phys_weights)
        self.opt_net.set_weights(self.opt_net_weights)
    def get_op_weigth(self):
        print(self.opt_phys.get_config())
        return self.opt_phys.variables(),self.opt_net.variables()
    def get_lrnet(self):
        return self.opt_net.learning_rate
    def get_lrphys(self):
        return self.opt_phys.learning_rate
