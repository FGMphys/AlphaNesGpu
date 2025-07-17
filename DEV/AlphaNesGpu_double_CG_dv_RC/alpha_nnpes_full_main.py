import os
import time
import sys

import logging
import absl.logging

# Disabilita tutti i messaggi di avvertimento per il modulo absl
absl.logging.set_verbosity("ERROR")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import yaml
import pickle
import glob as gl

from numpy.random import seed
from numpy import random
from numpy.random import default_rng


from source_routine.descriptor_builder import descriptor_layer

from optimizer_learning_rate_utility import build_learning_rate
from optimizer_learning_rate_utility import build_optimizer
from init_params.init_AFs_param import init_AFs_param



print("\n RUNNING ON TF VERSION ",tf.__version__)
try:
   numthreads=int(os.environ['TF_INTER_THREADS'])
   print("alpha_nes: tensorflow inter threads set to work with %d threads"%numthreads)
except:
   numthreads=1
#   print("alpha_nes: tensorflow set to work with %d threads"%numthreads)
tf.config.threading.set_inter_op_parallelism_threads(numthreads)
print("alpha_nes: tensorflow inter threads set to work with %d threads"%tf.config.threading.get_inter_op_parallelism_threads())
try:
   numthreads=int(os.environ['TF_INTRA_THREADS'])
except:
   numthreads=1
tf.config.threading.set_intra_op_parallelism_threads(numthreads)
print("alpha_nes: tensorflow intra threads set to work with %d threads"%tf.config.threading.get_intra_op_parallelism_threads())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

@tf.function()
def MSE(ypred,y):
   loss_function=tf.reduce_mean(tf.square((ypred-y)))
   return loss_function

def make_dataset_stream(base_pattern,mode):
    energy_on_disk=np.load(base_pattern+'/'+mode+'/'+'energy.npy',mmap_mode='r')
    force_on_disk=np.load(base_pattern+'/'+mode+'/'+'force.npy',mmap_mode='r')

    pos_on_disk=np.load(base_pattern+'/'+mode+'/'+'pos.npy',mmap_mode='r')
    box_on_disk=np.load(base_pattern+'/'+mode+'/'+'box.npy',mmap_mode='r')


    return energy_on_disk,force_on_disk,pos_on_disk,box_on_disk

def check_dimension(buffdim,dimension,mode):
    res=buffdim
    if buffdim>dimension:
       print("alpha_nes: buffdim in ",mode," mode is bigger than number of frames in the dataset. We set buffdim=datasetdim!")
       res=dimension
    return res
def make_idx_str(dimension,buffdim,mode):
    buffdim=check_dimension(buffdim,dimension,mode)
    truedim=dimension//buffdim*buffdim
    rejected=dimension%buffdim
    print("\nalpha_nes: Dataset in mode ",mode," has frames ",dimension,"\n")
    print("\nalpha_nes: It will be rejected ",rejected,' frames picked randomly to ensure batch size and buffer requested.\n')
    vec=np.arange(0,dimension)
    np.random.shuffle(vec)
    vec=np.reshape(vec[:truedim],(dimension//buffdim,buffdim))
    if mode=='test':
       np.savetxt("shuffle_dataset_vec",vec)
    return buffdim,vec

def check_along_frames(list_of_arr,axis):
    ref=list_of_arr[0].shape[axis]
    for el in list_of_arr:
        if ref!=el.shape[axis]:
           sys.exit("Dataset are not valid. Error on dimension along axis "+str(axis))
    return 0

def make_typemap(tipos):
    num=0
    list_tmap=[]
    for el in tipos:
        for k in range(el):
            list_tmap.append(num)
        num=num+1
    return list_tmap

def read_cutoff_info(full_param):
    rs=float(full_param['Rs'])
    rc=float(full_param['Rc'])
    rc_ang_inter=float(full_param['Rc_Angular_Inter'])
    rc_inter=float(full_param['Rc_Inter'])
    rad_buff=int(full_param['Radial_Buffer'])
    rc_ang=float(full_param['Rc_Angular'])
    maxneigh=int(full_param['Max_Angular_Neigh'])
    ang_buff=int(maxneigh*(maxneigh-1)/2)
    print("alpha_nes: Rc ",rc," Radial_Buffer ",rad_buff," Rc_Angular ",
           rc_ang,"Angular_Buffer ",ang_buff,"Hard cut-off ",rs)
    return [rc,rad_buff,rc_ang,ang_buff,rs,rs_inter,rc_ang_inter,rc_inter]

def order_folder(x):
    try:
        res=int(x.split('log')[-1])
    except:
        res=-1
    return res

def make_loss(full_param):
    try:
        loss_meth=full_param['loss_method']
        if loss_meth=='huber':
           HUBER = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
           model_loss=HUBER
           val_loss=MSE
           print("alpha_nes: the loss function is huber loss and validation loss is MSE")
        else:
           model_loss=MSE
           val_loss=MSE
           print("alpha_nes: the loss function is MSE loss as the validation loss")
    except:
        HUBER = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        model_loss=HUBER
        val_loss=MSE
        print("alpha_nes: the loss function is huber loss and validation loss is MSE")
    try:
        pe=tf.constant(float(full_param['loss_energy_prefactor']),dtype='float64')
        pf=tf.constant(float(full_param['loss_force_prefactor']),dtype='float64')
        pb=tf.constant(1.,dtype='float64')
        print("alpha_nes: pe and pf set to custom values",pe.numpy(),pf.numpy(),sep=' ',end='\n')
    except:
        pe=tf.constant(1.,dtype='float64')
        pf=tf.constant(1.,dtype='float64')
        pb=tf.constant(1.,dtype='float64')
        print("alpha_nes: pe and pf set to default value 1 1",sep=' ',end='\n')

    return model_loss,val_loss,pe,pf,pb

def make_method(full_param,model):
    try:
       train_meth=full_param['type_of_training']
    except:
       train_meth='energy+force'
    if train_meth=='energy+force':
       trainmeth=model.full_train_e_f
       testmeth=model.full_test_e_f
       print("alpha_nes: training will be on both energies and forces")
    elif train_meth=='energy':
         trainmeth=model.full_train_e
         testmeth=model.full_test_e
         print("alpha_nes: training will be on  energies only")
    else:
        sys.exit("alpha_nes: Error in type_of_training key. Possible choices are energy+force or energy")
    return trainmeth,testmeth




##Read the input file
with open(sys.argv[1]) as file:
    full_param = yaml.load(file, Loader=yaml.FullLoader)
base_pattern=full_param['dataset_folder']
"""
try:
    tipos=np.loadtxt(base_pattern+"/type.dat",dtype='int').reshape(-1,1)
    if tipos.shape[0]>1:
       tipos=[n_per_type for n_per_type in tipos[:,0]]
       type_map=make_typemap(tipos)
       np.savetxt('type_map.dat',np.array(type_map,dtype='int'),fmt='%d')
    else:
       tipos=[tipos[0,0]]
       type_map=make_typemap(tipos)
       np.savetxt('type_map.dat',np.array(type_map,dtype='int'),fmt='%d')
    nt=len(tipos)
    print("alpha_nes: detected ",nt," types of atoms.")
    N=len(type_map)
except:
    sys.exit("alpha_nes: In the dataset folder it is expected to have a type.dat file with the code for the atom type!")
"""
try:
    color_type_map=np.loadtxt(base_pattern+"/color_type_map.dat",dtype='int32').reshape(-1,1)
    Number_of_particles=color_type_map.shape[0]
except:
    sys.exit("alpha_nes: In the dataset folder it is expected to have a color_type_map.dat file with the color code for each site!")



from gradient_utility.mixture import register_force_3bAFs_grad
from gradient_utility.mixture import register_force_2bAFs_grad
from gradient_utility.mixture import register_3bAFs_grad
from gradient_utility.mixture import register_2bAFs_grad

from alphanes_models.mixture.alpha_nes_model import alpha_nes_full

from source_routine.mixture.physics_layer_mod import physics_layer
from source_routine.mixture.physics_layer_mod import lognorm_layer
from source_routine.mixture.force_layer_mod import force_layer




################# MAIN #########################################################
#Set seed
try:
    seed_par=int(full_param['Seed'])
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    os.environ['PYTHONHASHSEED']=str(seed_par)
    print("alpha_nes: seed fixed to custom value ", seed_par,end='\n')
except:
    seed_par=12345
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    os.environ['PYTHONHASHSEED']=str(seed_par)
    print("alpha_nes: seed fixed by default 12345\n")
#Read dataset map on disk
[e_map_tr,f_map_tr,pos_map_tr,box_map_tr]=make_dataset_stream(base_pattern,'training')
[e_map_ts,f_map_ts,pos_map_ts,box_map_ts]=make_dataset_stream(base_pattern,'test')
map_intra=np.loadtxt(full_param['map_intra_file'],dtype='int32')
###Check dimension of dataset
check_along_frames([e_map_tr,f_map_tr,pos_map_tr,box_map_tr],0)
check_along_frames([e_map_ts,f_map_ts,pos_map_ts,box_map_ts],0)
#Building a stream vector
buffer_stream_tr=full_param['buffer_stream_dim_tr']
buffer_stream_ts=full_param['buffer_stream_dim_ts']

subsamp=full_param['subsampling']
if subsamp!='no':
   dimtr=int(subsamp.split()[0])
   dimts=int(subsamp.split()[1])
else:
   dimtr=pos_map_tr.shape[0]
   dimts=pos_map_ts.shape[0]
[buffer_stream_tr,idx_str_tr]=make_idx_str(dimtr,buffer_stream_tr,'train')
[buffer_stream_ts,idx_str_ts]=make_idx_str(dimts,buffer_stream_ts,'test')


### Loop parameters
ne=int(full_param['number_of_epochs'])

bs=int(full_param['batch_size'])
if ((buffer_stream_tr%bs)!=0.):
   sys.exit("alpha_nes: batch size must be a divisor of buffer stream train dimension")
else:
   print("alpha_nes: batch selected for train is ",bs)
bs_test=int(full_param['batch_size_test'])
if ((buffer_stream_ts%bs_test)!=0.):
   sys.exit("alpha_nes: batch size must be a divisor of buffer stream test dimension")
else:
   print("alpha_nes: batch selected for test is ",bs_test)

#nb=idx_str_tr.shape[1]//bs+idx_str_tr.shape[1]%bs
nb=int(buffer_stream_tr/bs)

#Fix precision
tf.keras.backend.set_floatx('float64')

##Building the learning rate and then the optimizer
try:
    restart_par=full_param['restart']
    if os.path.isdir(restart_par):
        print("alpha_nes: Training will restart from state of folder ",restart_par)
        print("alpha_nes: Be sure of using the same input file of previous run")
    elif restart_par=='from_last':
        folders=gl.glob('model_log*')
        folders.sort(key=order_folder)
        restart_par=folders[-2]
        print("alpha_nes: Training will restart from last previous state ",restart_par)
    elif restart_par=='only_afs':
        print("alpha_nes: AFs will be initialised by user. Be sure to have defined params_folder key.")
    elif restart_par=='all_params':
        print("alpha_nes: AFs and NN parameters will be initialised by user. Be sure to have defined params_folder key.")
        restart_par=full_param['params_folder']
    else:
        restart_par='no'
        print("alpha_nes: Not indicated or not existing restart folder. It will be begun a new run")
except:
    restart_par='no'
    print("alpha_nes: Not indicated a correct restart command or not existing restart folder. It will be begun a new run")
restart=restart_par

##If we are not restarting, we initialiaze the optimizer and the learning rate
if full_param['restart'] in ['no','only_afs','all_params']:
    print("alpha_nes: Not previous optimizer state point will be loaded since restart_par ",restart_par," has been selected")
    lr_net_param=full_param['lr_dense_net'].split()
    lr_net=build_learning_rate(lr_net_param,ne,nb,idx_str_tr.shape[0],'net',0)

    opt_net_param=full_param['optimizer_net'].split()
    opt_net=build_optimizer(opt_net_param,lr_net,0)

    lr_phys_param=full_param['lr_phys_net'].split()
    lr_phys=build_learning_rate(lr_phys_param,ne,nb,idx_str_tr.shape[0],'phys',0)
##else we load the internal state of optimizer at the given point of previous training
else:
    with open(restart+'/opt_net_conf','rb') as source:
         config_net=pickle.load(source)
    opt_net=tf.keras.optimizers.Adam()
    opt_net=opt_net.from_config(config_net)

##Here we fix the value that prevents the explosion of the exponential
try:
    alpha_bound=float(full_param['alpha_bound'])
    print("alpha_nes: alphas will be upper-bound to custom",alpha_bound,sep=' ',end='\n')
except:
    alpha_bound=1.
    print("alpha_nes: alphas will be upper-bound to default",alpha_bound,sep=' ',end='\n')

#nt=len(tipos)
#nt_couple=int(nt+nt*(nt-1)/2)

#Initializing params for atomic finger prints
rng_state = np.random.get_state()
number_of_interaction=3 #intra, inert, sticky
map_rad_afs=full_param['map_rad_afs']
number_of_NN=len(map_rad_afs)
[init_alpha2b,init_alpha3b,init_mu,initial_type_emb,new_rng_state]=init_AFs_param(restart_par,full_param,number_of_interaction,rng_state)
np.random.set_state(new_rng_state)
#Reading cutoff info from input file
[rc,rad_buff,rc_ang,ang_buff,Rs,rc_ang_inter,rc_inter]=read_cutoff_info(full_param)
#################INITIALISE ALL THE LAYER FOR THE MODEL ##############################
#######Initialise Descriptor Layer###################################################
max_batch=int(np.max([buffer_stream_tr,buffer_stream_ts]))
Descriptor_Layer=descriptor_layer(rc,rad_buff,rc_ang,ang_buff,Number_of_particles,box_map_tr[0],Rs,max_batch,rc_ang_inter,rc_inter)
#######Initialise AFS Layer

Physics_Layers=[physics_layer(init_alpha2b[num_type],init_alpha3b[num_type],
                                initial_type_emb[num_type]) for num_type
                                in range(number_of_NN)]
##Initialise Log layer
Lognorm_Layers=[lognorm_layer(init_mu[num_type]) for num_type in range(number_of_NN)]
##Initialise force layer
Force_Layer=force_layer(rad_buff,ang_buff)
########Define Loss
[model_loss,val_loss,pe,pf,pb]=make_loss(full_param)
###Compose the model by concatenation of layers
model=alpha_nes_full(Physics_Layers,Force_Layer,1,model_loss,
             val_loss,opt_net,alpha_bound,Lognorm_Layers,
             color_type_map,restart,seed_par,full_param)
[trainmeth,testmeth]=make_method(full_param,model)
#################################################################################
#################################################################################

bestval=10**5
if restart_par!='no' and restart_par!='only_afs':
   fileOU=open('lcurve.out','a')
   print("alpha_nes: learning curve restart from ",restart_par)
   out_time=open("time_story_restart.dat",'a')
   lr_file=open("lr_step.dat",'a')
else:
   fileOU=open('lcurve.out','w')
   print("#num_step #num_epoch #RMSE_e   #RMSE_f   #Loss_Tot   #lr_net\n",file=fileOU)
   out_time=open("time_story.dat",'w')
   print("#Time per epoch training  #Time per epoch test\n",file=out_time)
   lr_file=open("lr_step.dat",'w')

model_name=full_param['model_name']
if full_param['restart']  in ['no','only_afs','all_params']:
    restart_ep=0
    os.mkdir(model_name)
    model.save_model_init(model_name)
    print("alpha_nes: Optimizer state will be initialized from zero")
    for k in range(number_of_NN):
       Physics_Layers[k].savealphas(model_name,"type"+str(k)+"initial_")
       Lognorm_Layers[k].savemu(model_name,"type"+str(k)+"initial_")
       accumul=0
else:
    restart_ep=int(restart_par.split('log')[-1])+1
    accumul=restart_ep*nb*idx_str_tr.shape[0]
    model_name=model_name
    index=np.arange(0,bs)
    [raddescr,angdescr,des3bsupp,
    intmap2b,intmap3b,intder2b,
    intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(tf.constant(pos_map_tr[index]),tf.constant(box_map_tr[index]),tf.constant(map_intra))
    k=0
    [dummyloss,dummylosse,dummylossb,dummylossf]=trainmeth(raddescr[k*bs:(k+1)*bs],angdescr[k*bs:(k+1)*bs],des3bsupp[k*bs:(k+1)*bs],intmap2b[k*bs:(k+1)*bs],intder2b[k*bs:(k+1)*bs],intmap3b[k*bs:(k+1)*bs],intder3b[k*bs:(k+1)*bs],intder3bsupp[k*bs:(k+1)*bs],numtriplet[k*bs:(k+1)*bs],e_map_tr[index][k*bs:(k+1)*bs],f_map_tr[index][k*bs:(k+1)*bs],0.,0.,0.)
    model.build_opt_weights()
    model.set_opt_weight()


lcurve_notmean=open('lcurve_notmean','w')
try:
   displ_freq=int(full_param['displ_freq'])
except:
   displ_freq=1
try:
   freq_test=int(full_param['freq_test'])
   print("alpha_nes: test will be ever ",freq_test," epochs")
except:
   freq_test=1
   print("alpha_nes: test will be ever ",freq_test," epochs")
start_loc=time.time()
for ep in range(restart_ep,ne):
    losstot=tf.constant(0.,dtype='float64')
    vallosstot=tf.constant(0.,dtype='float64')
    vallosstote=tf.constant(0.,dtype='float64')
    vallosstotf=tf.constant(0.,dtype='float64')
    for numbuf,el in enumerate(idx_str_tr):
        loss_buffer=0.
        start=time.time()
        [raddescr,angdescr,des3bsupp,
        intmap2b,intmap3b,intder2b,
        intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(tf.constant(pos_map_tr[el]),tf.constant(box_map_tr[el]),tf.constant(map_intra))
        max_ang=np.max(numtriplet.numpy())
        max_buff=int(max_ang*(max_ang-1)/2)
        if (max_buff>ang_buff):
            print("alpha_nes: found angular neighbours beyond the buffer (%d vs %d)"%(max_buff,ang_buff))
            sys.exit()
        nb=int(buffer_stream_tr/bs)
        for k in range(nb):
            start3=time.time()
            [loss,losse,loss_bound,lossf]=trainmeth(raddescr[k*bs:(k+1)*bs],angdescr[k*bs:(k+1)*bs],des3bsupp[k*bs:(k+1)*bs],intmap2b[k*bs:(k+1)*bs],intder2b[k*bs:(k+1)*bs],intmap3b[k*bs:(k+1)*bs],intder3b[k*bs:(k+1)*bs],intder3bsupp[k*bs:(k+1)*bs],numtriplet[k*bs:(k+1)*bs],e_map_tr[el][k*bs:(k+1)*bs],f_map_tr[el][k*bs:(k+1)*bs],pe,pf,pb)
            lrnow=model.get_lrnet()
            print(losse.numpy(),lossf.numpy(),loss_bound.numpy(),file=lcurve_notmean)
            lcurve_notmean.flush()
            lr_file.write(str(lrnow.numpy())+'\n')
            lr_file.flush()
            accumul=accumul+1
            loss_buffer+=loss
        losstot+=loss_buffer
        if accumul%displ_freq==0:
           print("Epoch ",ep," step ",accumul,". Time to elaborate ",displ_freq," batch of ",bs," frames is",(time.time()-start_loc))
           print("Epoch ",ep," step ",accumul,". Time to elaborate ",displ_freq," batch of ",bs," frames is",(time.time()-start_loc),file=out_time)
           start_loc=time.time()
    losstot*=1/(k+1)/(numbuf+1)
    stop_tr=time.time()
    lcurve_notmean.flush()
    if (ep%freq_test==0):
       for numbuf,el in enumerate(idx_str_ts):
           vallosstot_buff=0.
           vallosstote_buff=0.
           vallosstotf_buff=0.
           [raddescr,angdescr,des3bsupp,
           intmap2b,intmap3b,intder2b,
           intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(tf.constant(pos_map_ts[el]),tf.constant(box_map_ts[el]),tf.constant(map_intra))
           nb=int(buffer_stream_ts/bs_test)
           for k in range(nb):
               [val_loss,val_lossf,val_losse]=testmeth(raddescr[k*bs_test:(k+1)*bs_test],angdescr[k*bs_test:(k+1)*bs_test],des3bsupp[k*bs_test:(k+1)*bs_test],intmap2b[k*bs_test:(k+1)*bs_test],intder2b[k*bs_test:(k+1)*bs_test],intmap3b[k*bs_test:(k+1)*bs_test],intder3b[k*bs_test:(k+1)*bs_test],intder3bsupp[k*bs_test:(k+1)*bs_test],numtriplet[k*bs_test:(k+1)*bs_test],e_map_ts[el][k*bs_test:(k+1)*bs_test],f_map_ts[el][k*bs_test:(k+1)*bs_test])
               vallosstot_buff+=val_loss
               vallosstote_buff+=val_losse
               vallosstotf_buff+=val_lossf
           vallosstot+=vallosstot_buff
           vallosstote+=vallosstote_buff
           vallosstotf+=vallosstotf_buff
       vallosstot=vallosstot/(k+1)/(numbuf+1)
       vallosstote=vallosstote/(k+1)/(numbuf+1)
       vallosstotf=vallosstotf/(k+1)/(numbuf+1)


       outfold_name=model_name+str(ep)
       model.save_model(outfold_name)
       np.savetxt(outfold_name+"/model_error",[np.sqrt(vallosstote),np.sqrt(vallosstotf)],header='RMSE_e  RMSE_f ')
       print(accumul,ep,np.sqrt(vallosstote.numpy()),(np.sqrt(vallosstotf.numpy())),losstot.numpy(),lrnow.numpy(),sep=' ',end='\n',file=fileOU)
       print("Testing model at global step",accumul," and epoch ",ep," val_lossE ",np.sqrt(vallosstote.numpy())," val_lossF ",np.sqrt(vallosstotf.numpy())," loss_Tot ",losstot.numpy()," lr_net ",lrnow.numpy(),sep=' ',end='\n')
       print("We are at epoch ",ep)
       fileOU.flush()
       out_time.flush()
