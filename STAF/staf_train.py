import os
import time
import sys
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

_STAF_HOME = Path(__file__).resolve().parent
if str(_STAF_HOME) not in sys.path:
    sys.path.insert(0, str(_STAF_HOME))
from staf.dtype import set_precision, tf_dtype, zero  # noqa: E402

from optimizer_learning_rate_utility import build_learning_rate
from optimizer_learning_rate_utility import build_optimizer
from init_params.init_AFs_param import init_AFs_param
from learning_utility.metrics_log import MetricsLog



print("\n RUNNING ON TF VERSION ",tf.__version__)
try:
   numthreads=int(os.environ['TF_INTER_THREADS'])
   print("STAF: tensorflow inter threads set to work with %d threads"%numthreads)
except:
   numthreads=1
#   print("STAF: tensorflow set to work with %d threads"%numthreads)
tf.config.threading.set_inter_op_parallelism_threads(numthreads)
print("STAF: tensorflow inter threads set to work with %d threads"%tf.config.threading.get_inter_op_parallelism_threads())
try:
   numthreads=int(os.environ['TF_INTRA_THREADS'])
except:
   numthreads=1
tf.config.threading.set_intra_op_parallelism_threads(numthreads)
print("STAF: tensorflow intra threads set to work with %d threads"%tf.config.threading.get_intra_op_parallelism_threads())

def _configure_gpu_memory_growth(gpus):
    if not gpus:
        print("STAF: no GPU detected")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def init_distribute(full_param):
    """Return (mode, strategy|None, hvd_module|None).

    GPU memory growth / visibility is configured here (after YAML) so Horovod
    can pin each MPI rank to gpus[local_rank] before TF initializes devices.
    """
    mode = str(full_param.get('distribute', 'none')).strip().lower()
    if mode in ('', 'none', 'null', 'single'):
        _configure_gpu_memory_growth(tf.config.list_physical_devices('GPU'))
        print("STAF: distribute=none (single device)")
        return 'none', None, None

    if mode == 'horovod':
        try:
            import horovod.tensorflow as hvd
        except ImportError:
            sys.exit(
                "STAF: distribute=horovod requires the horovod package.\n"
                "  Install (example): HOROVOD_WITH_TENSORFLOW=1 pip install horovod\n"
                "  Launch: mpirun -np <N> python staf_train.py <input.yaml>"
            )
        hvd.init()
        gpus = tf.config.list_physical_devices('GPU')
        _configure_gpu_memory_growth(gpus)
        if gpus:
            if hvd.local_rank() >= len(gpus):
                sys.exit(
                    "STAF: horovod local_rank=%d but only %d GPU(s) visible"
                    % (hvd.local_rank(), len(gpus))
                )
            tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        print(
            "STAF: distribute=horovod size=%d rank=%d local_rank=%d"
            % (hvd.size(), hvd.rank(), hvd.local_rank())
        )
        return 'horovod', None, hvd

    if mode != 'mirrored':
        sys.exit(
            "STAF: unknown distribute=%r (use none | mirrored | horovod)" % mode
        )

    _configure_gpu_memory_growth(tf.config.list_physical_devices('GPU'))
    devices_yaml = full_param.get('devices', None)
    if devices_yaml is not None:
        # devices: optional list of logical GPU indices (ignored for horovod).
        dev_names = ['/GPU:%d' % int(i) for i in devices_yaml]
        strategy = tf.distribute.MirroredStrategy(devices=dev_names)
        print(
            "STAF: distribute=mirrored replicas=%d devices=%s"
            % (strategy.num_replicas_in_sync, dev_names)
        )
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(
            "STAF: distribute=mirrored replicas=%d devices=all_visible"
            % strategy.num_replicas_in_sync
        )
    return 'mirrored', strategy, None

def wrap_train_method(trainmeth, strategy):
    """Call trainmeth under strategy.run; unwrap PerReplica for host logging.

    With 1 replica, skip strategy.run so Keras losses keep SUM_OVER_BATCH_SIZE
    (allowed only outside Strategy.run). Multi-replica uses SUM losses + mean.
    Horovod uses DistributedOptimizer instead (no strategy.run).
    """
    if strategy is None or strategy.num_replicas_in_sync <= 1:
        return trainmeth

    @tf.function(reduce_retracing=True)
    def _distributed(*args):
        return strategy.run(trainmeth, args=args)

    def _call(*args):
        out = _distributed(*args)
        unwrapped = []
        for x in out:
            parts = strategy.experimental_local_results(x)
            if len(parts) == 1:
                unwrapped.append(parts[0])
            else:
                unwrapped.append(
                    tf.add_n(list(parts)) / float(len(parts))
                )
        return tuple(unwrapped)

    return _call

def scale_lr_param_for_horovod(lr_param_split, hvd_mod):
    """Scale initial LR by hvd.size() (global-batch convention)."""
    if hvd_mod is None or hvd_mod.size() <= 1:
        return lr_param_split
    out = list(lr_param_split)
    try:
        out[1] = str(float(out[1]) * float(hvd_mod.size()))
        print(
            "STAF: horovod scaled initial LR by size=%d → %s"
            % (hvd_mod.size(), out[1])
        )
    except (IndexError, ValueError):
        pass
    return out

def shard_idx_str(idx_str, hvd_mod, name):
    """Frame-buffer sharding across MPI ranks (train path)."""
    if hvd_mod is None or hvd_mod.size() <= 1:
        return idx_str
    shard = idx_str[hvd_mod.rank()::hvd_mod.size()]
    if shard.shape[0] == 0:
        sys.exit(
            "STAF: horovod shard empty for %s on rank %d (size=%d); "
            "use fewer ranks or a larger dataset"
            % (name, hvd_mod.rank(), hvd_mod.size())
        )
    print(
        "STAF: horovod sharded %s buffers %d → %d (rank %d/%d)"
        % (name, idx_str.shape[0], shard.shape[0], hvd_mod.rank(), hvd_mod.size())
    )
    return shard

def collect_broadcast_variables(model):
    """Model + optimizer variables for hvd.broadcast_variables."""
    vars_ = []
    for net in model.nets:
        vars_.extend(list(net.variables))
    for phys in model.physics_layer:
        vars_.extend([phys.alpha2b, phys.alpha3b, phys.type_emb_2b, phys.type_emb_3b])
    for logn in model.lognorm_layer:
        vars_.append(logn.mu)
    try:
        vars_.extend(list(model.opt_net.variables()))
    except Exception:
        pass
    try:
        vars_.extend(list(model.opt_phys.variables()))
    except Exception:
        pass
    return vars_

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
       print("STAF: buffdim in ",mode," mode is bigger than number of frames in the dataset. We set buffdim=datasetdim!")
       res=dimension
    return res
def make_idx_str(dimension,buffdim,mode,save_shuffle=True):
    buffdim=check_dimension(buffdim,dimension,mode)
    truedim=dimension//buffdim*buffdim
    rejected=dimension%buffdim
    print("\nSTAF: Dataset in mode ",mode," has frames ",dimension,"\n")
    print("\nSTAF: It will be rejected ",rejected,' frames picked randomly to ensure batch size and buffer requested.\n')
    vec=np.arange(0,dimension)
    np.random.shuffle(vec)
    vec=np.reshape(vec[:truedim],(dimension//buffdim,buffdim))
    if mode=='test' and save_shuffle:
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
    rad_buff=int(full_param['Radial_Buffer'])
    rc_ang=float(full_param['Rc_Angular'])
    maxneigh=int(full_param['Max_Angular_Neigh'])
    ang_buff=int(maxneigh*(maxneigh-1)/2)
    print("STAF: Rc ",rc," Radial_Buffer ",rad_buff," Rc_Angular ",
           rc_ang,"Angular_Buffer ",ang_buff,"Hard cut-off ",rs)
    return [rc,rad_buff,rc_ang,ang_buff,rs]

def order_folder(x):
    try:
        res=int(x.split('log')[-1])
    except:
        res=-1
    return res

def make_loss(full_param, distribute_mode='none', n_replicas=1):
    # MirroredStrategy.run forbids SUM_OVER_BATCH_SIZE; use SUM only with ≥2 GPUs.
    if distribute_mode == 'mirrored' and n_replicas > 1:
        huber_red = tf.keras.losses.Reduction.SUM
    else:
        huber_red = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    try:
        loss_meth=full_param['loss_method']
        if loss_meth=='huber':
           HUBER = tf.keras.losses.Huber(reduction=huber_red)
           model_loss=HUBER
           val_loss=MSE
           print("STAF: the loss function is huber loss and validation loss is MSE")
        else:
           model_loss=MSE
           val_loss=MSE
           print("STAF: the loss function is MSE loss as the validation loss")
    except:
        HUBER = tf.keras.losses.Huber(reduction=huber_red)
        model_loss=HUBER
        val_loss=MSE
        print("STAF: the loss function is huber loss and validation loss is MSE")
    dt = tf_dtype()
    try:
        pe=tf.constant(float(full_param['loss_energy_prefactor']),dtype=dt)
        pf=tf.constant(float(full_param['loss_force_prefactor']),dtype=dt)
        pb=tf.constant(1.,dtype=dt)
        print("STAF: pe and pf set to custom values",pe.numpy(),pf.numpy(),sep=' ',end='\n')
    except:
        pe=tf.constant(1.,dtype=dt)
        pf=tf.constant(1.,dtype=dt)
        pb=tf.constant(1.,dtype=dt)
        print("STAF: pe and pf set to default value 1 1",sep=' ',end='\n')

    return model_loss,val_loss,pe,pf,pb

def make_method(full_param,model):
    try:
       train_meth=full_param['type_of_training']
    except:
       train_meth='energy+force'
    if train_meth=='energy+force':
       trainmeth=model.full_train_e_f
       testmeth=model.full_test_e_f
       print("STAF: training will be on both energies and forces")
    elif train_meth=='energy':
         trainmeth=model.full_train_e
         testmeth=model.full_test_e
         print("STAF: training will be on  energies only")
    else:
        sys.exit("STAF: Error in type_of_training key. Possible choices are energy+force or energy")
    return trainmeth,testmeth




##Read the input file
with open(sys.argv[1]) as file:
    full_param = yaml.load(file, Loader=yaml.FullLoader)
set_precision(full_param.get("precision"))
distribute_mode, strategy, hvd_mod = init_distribute(full_param)
is_chief = (hvd_mod is None) or (hvd_mod.rank() == 0)

def dist_scope():
    """Fresh strategy.scope() each time (context managers are single-use)."""
    return strategy.scope() if strategy is not None else contextlib.nullcontext()

base_pattern=full_param['dataset_folder']
try:
    tipos=np.loadtxt(base_pattern+"/type.dat",dtype='int').reshape(-1,1)
    if tipos.shape[0]>1:
       tipos=[n_per_type for n_per_type in tipos[:,0]]
       type_map=make_typemap(tipos)
       if is_chief:
           np.savetxt('type_map.dat',np.array(type_map,dtype='int'),fmt='%d')
    else:
       tipos=[tipos[0,0]]
       type_map=make_typemap(tipos)
       if is_chief:
           np.savetxt('type_map.dat',np.array(type_map,dtype='int'),fmt='%d')
    nt=len(tipos)
    print("STAF: detected ",nt," types of atoms.")
    N=len(type_map)
except:
    sys.exit("STAF: In the dataset folder it is expected to have a type.dat file with the code for the atom type!")


from gradient_utility import register_force_3bAFs_grad
from gradient_utility import register_force_2bAFs_grad
from gradient_utility import register_3bAFs_grad
from gradient_utility import register_2bAFs_grad

from staf_models.staf_model import staf_full
from source_routine.descriptor_builder import descriptor_layer

from source_routine.physics_layer_mod import physics_layer
from source_routine.physics_layer_mod import lognorm_layer
from source_routine.force_layer_mod import force_layer






################# MAIN #########################################################
#Set seed
try:
    seed_par=int(full_param['Seed'])
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    os.environ['PYTHONHASHSEED']=str(seed_par)
    print("STAF: seed fixed to custom value ", seed_par,end='\n')
except:
    seed_par=12345
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    os.environ['PYTHONHASHSEED']=str(seed_par)
    print("STAF: seed fixed by default 12345\n")
#Read dataset map on disk
[e_map_tr,f_map_tr,pos_map_tr,box_map_tr]=make_dataset_stream(base_pattern,'training')
[e_map_ts,f_map_ts,pos_map_ts,box_map_ts]=make_dataset_stream(base_pattern,'test')
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
[buffer_stream_tr,idx_str_tr]=make_idx_str(dimtr,buffer_stream_tr,'train',
                                             save_shuffle=is_chief)
[buffer_stream_ts,idx_str_ts]=make_idx_str(dimts,buffer_stream_ts,'test',
                                             save_shuffle=is_chief)
# Horovod: shard train buffers across ranks (test stays full; only rank 0 evaluates).
idx_str_tr = shard_idx_str(idx_str_tr, hvd_mod, 'train')

### Loop parameters
ne=int(full_param['number_of_epochs'])

bs=int(full_param['batch_size'])
if ((buffer_stream_tr%bs)!=0.):
   sys.exit("STAF: batch size must be a divisor of buffer stream train dimension")
else:
   print("STAF: batch selected for train is ",bs)
bs_test=int(full_param['batch_size_test'])
if ((buffer_stream_ts%bs_test)!=0.):
   sys.exit("STAF: batch size must be a divisor of buffer stream test dimension")
else:
   print("STAF: batch selected for test is ",bs_test)

#nb=idx_str_tr.shape[1]//bs+idx_str_tr.shape[1]%bs
nb=int(buffer_stream_tr/bs)

### Building Net parameters
actfun=full_param['activation_function']
nhl=full_param['number_of_decoding_layers']
if nhl>0:
   nD=[int(k) for k in full_param['number_of_decoding_nodes'].split()]
else:
   nD=0

# Precision already set from YAML (or inferred from this tree name)

##Building the learning rate and then the optimizer
try:
    restart_par=full_param['restart']
    if os.path.isdir(restart_par):
        print("STAF: Training will restart from state of folder ",restart_par)
        print("STAF: Be sure of using the same input file of previous run")
    elif restart_par=='from_last':
        folders=gl.glob('model_log*')
        folders.sort(key=order_folder)
        restart_par=folders[-2]
        print("STAF: Training will restart from last previous state ",restart_par)
    elif restart_par=='only_afs':
        print("STAF: AFs will be initialised by user. Be sure to have defined afs_param_folder key.")
    else:
        restart_par='no'
        print("STAF: Not indicated or not existing restart folder. It will be begun a new run")
except:
    restart_par='no'
    print("STAF: Not indicated or not existing restart folder. It will be begun a new run")
restart=restart_par

##If we are not restarting, we initialiaze the optimizer and the learning rate
## (under MirroredStrategy.scope when distribute=mirrored)
with dist_scope():
    if restart_par=='no' or restart_par=='only_afs':
        lr_net_param=scale_lr_param_for_horovod(
            full_param['lr_dense_net'].split(), hvd_mod)
        lr_net=build_learning_rate(lr_net_param,ne,nb,idx_str_tr.shape[0],'net',0)

        opt_net_param=full_param['optimizer_net'].split()
        opt_net=build_optimizer(opt_net_param,lr_net,0)

        lr_phys_param=scale_lr_param_for_horovod(
            full_param['lr_phys_net'].split(), hvd_mod)
        lr_phys=build_learning_rate(lr_phys_param,ne,nb,idx_str_tr.shape[0],'phys',0)
        opt_phys_param=full_param['optimizer_phys'].split()
        opt_phys=build_optimizer(opt_phys_param,lr_phys,0)
    ##else we load the internal state of optimizer at the given point of previous training
    else:
        with open(restart+'/opt_net_conf','rb') as source:
             config_net=pickle.load(source)
        opt_net=tf.keras.optimizers.Adam()
        opt_net=opt_net.from_config(config_net)
        with open(restart+'/opt_phys_conf','rb') as source:
             config_phys=pickle.load(source)
        opt_phys=tf.keras.optimizers.Adam()
        opt_phys=opt_phys.from_config(config_phys)
    if hvd_mod is not None:
        opt_net = hvd_mod.DistributedOptimizer(opt_net)
        opt_phys = hvd_mod.DistributedOptimizer(opt_phys)
        print("STAF: optimizers wrapped with hvd.DistributedOptimizer")

##Here we fix the value that prevents the explosion of the exponential
try:
    alpha_bound=float(full_param['alpha_bound'])
    print("STAF: alphas will be upper-bound to custom",alpha_bound,sep=' ',end='\n')
except:
    alpha_bound=1.
    print("STAF: alphas will be upper-bound to default",alpha_bound,sep=' ',end='\n')

limit=alpha_bound
limit3b=alpha_bound
nt=len(tipos)
nt_couple=int(nt+nt*(nt-1)/2)

#Initializing params for atomic finger prints
rng_state = np.random.get_state()
[init_alpha2b,init_alpha3b,init_mu,initial_type_emb,new_rng_state]=init_AFs_param(restart,full_param,nt,rng_state)
np.random.set_state(new_rng_state)
#Reading cutoff info from input file
[rc,rad_buff,rc_ang,ang_buff,Rs]=read_cutoff_info(full_param)
#################INITIALISE ALL THE LAYER FOR THE MODEL ##############################
#######Initialise Descriptor Layer###################################################
max_batch=int(np.max([buffer_stream_tr,buffer_stream_ts]))
Descriptor_Layer=descriptor_layer(rc,rad_buff,rc_ang,ang_buff,N,box_map_tr[0],Rs,max_batch)
########Define Loss (host-side constants; safe outside strategy.scope)
_n_replicas = strategy.num_replicas_in_sync if strategy is not None else 1
[model_loss,val_loss,pe,pf,pb]=make_loss(full_param, distribute_mode, _n_replicas)
### Layers + model under MirroredStrategy.scope when distribute=mirrored
with dist_scope():
    #######Initialise AFS Layer
    Physics_Layers=[physics_layer(init_alpha2b[num_type],init_alpha3b[num_type],
                                    initial_type_emb[num_type]) for num_type
                                    in range(nt)]
    ##Initialise Log layer
    Lognorm_Layers=[lognorm_layer(init_mu[num_type]) for num_type in range(nt)]
    ##Initialise force layer
    Force_Layer=force_layer(rad_buff,ang_buff)
    ###Compose the model by concatenation of layers
    model=staf_full(Physics_Layers,Force_Layer,nhl,nD,actfun,1,model_loss,
                 val_loss,opt_net,opt_phys,alpha_bound,Lognorm_Layers,tipos,
                 type_map,restart,seed_par)
[trainmeth,testmeth]=make_method(full_param,model)
trainmeth=wrap_train_method(trainmeth, strategy)
_hvd_need_bcast = (hvd_mod is not None)
#################################################################################
#################################################################################

bestval=10**5
_devnull = open(os.devnull, 'w')
if is_chief:
    if restart_par!='no' and restart_par!='only_afs':
       fileOU=open('lcurve.out','a')
       print("STAF: learning curve restart from ",restart_par)
       out_time=open("time_story_restart.dat",'a')
       lr_file=open("lr_step.dat",'a')
    else:
       fileOU=open('lcurve.out','w')
       # xmgrace: xmgrace -nxy lcurve.out
       print("# STAF epoch validation curve (test-set RMSE + training loss)", file=fileOU)
       print("# Columns: step RMSE_e RMSE_f Loss_Tot lr_net lr_phys epoch", file=fileOU)
       print("# Plot: xmgrace -nxy lcurve.out", file=fileOU)
       print("@    title \"STAF validation curve\"", file=fileOU)
       print("@    xaxis  label \"Global step\"", file=fileOU)
       print("@    yaxis  label \"RMSE / Loss\"", file=fileOU)
       print("@    s0 legend \"RMSE_e (test)\"", file=fileOU)
       print("@    s1 legend \"RMSE_f (test)\"", file=fileOU)
       print("@    s2 legend \"Loss_Tot (train)\"", file=fileOU)
       print("@    s3 legend \"lr_net\"", file=fileOU)
       print("@    s4 legend \"lr_phys\"", file=fileOU)
       print("@    s5 legend \"epoch\"", file=fileOU)
       out_time=open("time_story.dat",'w')
       print("#Time per epoch training  #Time per epoch test\n",file=out_time)
       lr_file=open("lr_step.dat",'w')
       print("# STAF learning-rate schedule (net)", file=lr_file)
       print("# Columns: lr_net", file=lr_file)
       print("@    title \"STAF lr_net\"", file=lr_file)
       print("@    s0 legend \"lr_net\"", file=lr_file)
else:
    fileOU = _devnull
    out_time = _devnull
    lr_file = _devnull

metrics_log = MetricsLog(full_param.get("metrics_log") if is_chief else None)

model_name=full_param['model_name']
if restart_par=='no' or restart_par=='only_afs' or restart_par=='all_params':
    restart_ep=0
    if is_chief:
        os.mkdir(model_name)
        model.save_model_init(model_name)
        for k in range(nt):
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
    intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(pos_map_tr[index],box_map_tr[index])
    k=0
    [dummyloss,dummylosse,dummylossb,dummylossf]=trainmeth(raddescr[k*bs:(k+1)*bs],angdescr[k*bs:(k+1)*bs],des3bsupp[k*bs:(k+1)*bs],intmap2b[k*bs:(k+1)*bs],intder2b[k*bs:(k+1)*bs],intmap3b[k*bs:(k+1)*bs],intder3b[k*bs:(k+1)*bs],intder3bsupp[k*bs:(k+1)*bs],numtriplet[k*bs:(k+1)*bs],e_map_tr[index][k*bs:(k+1)*bs],f_map_tr[index][k*bs:(k+1)*bs],0.,0.,0.)
    model.build_opt_weights()
    model.set_opt_weight()
    if _hvd_need_bcast:
        hvd_mod.broadcast_variables(collect_broadcast_variables(model), root_rank=0)
        _hvd_need_bcast = False
        print("STAF: horovod broadcast after restart warm-up (root=0)")


if is_chief:
    lcurve_notmean=open('lcurve_notmean','w')
    # xmgrace: xmgrace -nxy lcurve_notmean
    print("# STAF per-batch losses (not epoch-averaged)", file=lcurve_notmean)
    print("# Columns: step Loss_E Loss_F Loss_Bound", file=lcurve_notmean)
    print("# Plot: xmgrace -nxy lcurve_notmean", file=lcurve_notmean)
    print("@    title \"STAF batch losses (lcurve_notmean)\"", file=lcurve_notmean)
    print("@    xaxis  label \"Global step\"", file=lcurve_notmean)
    print("@    yaxis  label \"Batch loss\"", file=lcurve_notmean)
    print("@    s0 legend \"Loss_E\"", file=lcurve_notmean)
    print("@    s1 legend \"Loss_F\"", file=lcurve_notmean)
    print("@    s2 legend \"Loss_Bound\"", file=lcurve_notmean)
else:
    lcurve_notmean = _devnull
try:
   displ_freq=int(full_param['displ_freq'])
except:
   displ_freq=1
# Host sync (.numpy / flush) only every log_batch_freq steps (default: displ_freq).
try:
   log_batch_freq=int(full_param.get('log_batch_freq', displ_freq))
except Exception:
   log_batch_freq=displ_freq
if log_batch_freq < 1:
   log_batch_freq=1
print("STAF: batch loss/lr host log every", log_batch_freq, "steps")
try:
   freq_test=int(full_param['freq_test'])
   print("STAF: test will be ever ",freq_test," epochs")
except:
   freq_test=1
   print("STAF: test will be ever ",freq_test," epochs")

_dt = tf_dtype()
_np_dtype = np.float64 if _dt == tf.float64 else np.float32

def _load_buffer_host(pos_map, box_map, e_map, f_map, el):
    """Copy mmap slices on a host thread so GPU can train the previous buffer."""
    return (np.asarray(pos_map[el], dtype=_np_dtype),
            np.asarray(box_map[el], dtype=_np_dtype),
            np.asarray(e_map[el], dtype=_np_dtype),
            np.asarray(f_map[el], dtype=_np_dtype))

print("STAF: reduce_retracing on train/test + host buffer prefetch")
start_loc=time.time()
_prefetch_pool = ThreadPoolExecutor(max_workers=1)
for ep in range(restart_ep,ne):
    losstot=zero()
    vallosstot=zero()
    vallosstote=zero()
    vallosstotf=zero()
    nbuf_tr = idx_str_tr.shape[0]
    fut = _prefetch_pool.submit(
        _load_buffer_host, pos_map_tr, box_map_tr, e_map_tr, f_map_tr, idx_str_tr[0]
    ) if nbuf_tr else None
    for numbuf in range(nbuf_tr):
        loss_buffer=0.
        start=time.time()
        pos_np, box_np, e_np, f_np = fut.result()
        if numbuf + 1 < nbuf_tr:
            fut = _prefetch_pool.submit(
                _load_buffer_host, pos_map_tr, box_map_tr, e_map_tr, f_map_tr,
                idx_str_tr[numbuf + 1]
            )
        pos_t = tf.convert_to_tensor(pos_np, dtype=_dt)
        box_t = tf.convert_to_tensor(box_np, dtype=_dt)
        e_t = tf.convert_to_tensor(e_np, dtype=_dt)
        f_t = tf.convert_to_tensor(f_np, dtype=_dt)
        [raddescr,angdescr,des3bsupp,
        intmap2b,intmap3b,intder2b,
        intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(pos_t, box_t)
        # Host sync only once per epoch (first buffer) — avoids D2H every buffer.
        if numbuf == 0:
            max_ang=int(tf.reduce_max(numtriplet).numpy())
            max_buff=int(max_ang*(max_ang-1)/2)
            if (max_buff>ang_buff):
                print("STAF: found angular neighbours beyond the buffer (%d vs %d)"%(max_buff,ang_buff))
                sys.exit()
        nb=int(buffer_stream_tr/bs)
        for k in range(nb):
            start3=time.time()
            [loss,losse,loss_bound,lossf]=trainmeth(
                raddescr[k*bs:(k+1)*bs], angdescr[k*bs:(k+1)*bs],
                des3bsupp[k*bs:(k+1)*bs], intmap2b[k*bs:(k+1)*bs],
                intder2b[k*bs:(k+1)*bs], intmap3b[k*bs:(k+1)*bs],
                intder3b[k*bs:(k+1)*bs], intder3bsupp[k*bs:(k+1)*bs],
                numtriplet[k*bs:(k+1)*bs], e_t[k*bs:(k+1)*bs], f_t[k*bs:(k+1)*bs],
                pe, pf, pb)
            if _hvd_need_bcast:
                hvd_mod.broadcast_variables(
                    collect_broadcast_variables(model), root_rank=0)
                _hvd_need_bcast = False
                print("STAF: horovod broadcast after first train step (root=0)")
            lrnow=model.get_lrnet()
            lrnow2=model.get_lrphys()
            accumul=accumul+1
            loss_buffer+=loss
            if is_chief and accumul % log_batch_freq == 0:
                print(accumul, float(losse.numpy()), float(lossf.numpy()),
                      float(loss_bound.numpy()), file=lcurve_notmean)
                lr_file.write(str(float(lrnow.numpy())) + '\n')
            if is_chief and accumul % displ_freq == 0:
                lcurve_notmean.flush()
                lr_file.flush()
                print("Epoch ",ep," step ",accumul,". Time to elaborate ",displ_freq," batch of ",bs," frames is",(time.time()-start_loc))
                print("Epoch ",ep," step ",accumul,". Time to elaborate ",displ_freq," batch of ",bs," frames is",(time.time()-start_loc),file=out_time)
                start_loc=time.time()
        losstot+=loss_buffer
    losstot*=1/(k+1)/(numbuf+1)
    stop_tr=time.time()
    if is_chief:
        lcurve_notmean.flush()
        lr_file.flush()
    # Test + checkpoint: rank 0 only under Horovod (avoids racing model_log*).
    if is_chief and (ep%freq_test==0):
       nbuf_ts = idx_str_ts.shape[0]
       fut_ts = _prefetch_pool.submit(
           _load_buffer_host, pos_map_ts, box_map_ts, e_map_ts, f_map_ts, idx_str_ts[0]
       ) if nbuf_ts else None
       for numbuf in range(nbuf_ts):
           vallosstot_buff=0.
           vallosstote_buff=0.
           vallosstotf_buff=0.
           pos_np, box_np, e_np, f_np = fut_ts.result()
           if numbuf + 1 < nbuf_ts:
               fut_ts = _prefetch_pool.submit(
                   _load_buffer_host, pos_map_ts, box_map_ts, e_map_ts, f_map_ts,
                   idx_str_ts[numbuf + 1]
               )
           pos_t = tf.convert_to_tensor(pos_np, dtype=_dt)
           box_t = tf.convert_to_tensor(box_np, dtype=_dt)
           e_t = tf.convert_to_tensor(e_np, dtype=_dt)
           f_t = tf.convert_to_tensor(f_np, dtype=_dt)
           [raddescr,angdescr,des3bsupp,
           intmap2b,intmap3b,intder2b,
           intder3b,intder3bsupp,numtriplet]=Descriptor_Layer(pos_t, box_t)
           nb=int(buffer_stream_ts/bs_test)
           for k in range(nb):
               [val_loss,val_lossf,val_losse]=testmeth(
                   raddescr[k*bs_test:(k+1)*bs_test], angdescr[k*bs_test:(k+1)*bs_test],
                   des3bsupp[k*bs_test:(k+1)*bs_test], intmap2b[k*bs_test:(k+1)*bs_test],
                   intder2b[k*bs_test:(k+1)*bs_test], intmap3b[k*bs_test:(k+1)*bs_test],
                   intder3b[k*bs_test:(k+1)*bs_test], intder3bsupp[k*bs_test:(k+1)*bs_test],
                   numtriplet[k*bs_test:(k+1)*bs_test],
                   e_t[k*bs_test:(k+1)*bs_test], f_t[k*bs_test:(k+1)*bs_test])
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
       rmse_e=float(np.sqrt(vallosstote.numpy()))
       rmse_f=float(np.sqrt(vallosstotf.numpy()))
       loss_tot_v=float(losstot.numpy())
       lr_net_v=float(lrnow.numpy())
       lr_finger_v=float(lrnow2.numpy())
       np.savetxt(outfold_name+"/model_error",[rmse_e,rmse_f],header='RMSE_e  RMSE_f ')
       print(accumul,rmse_e,rmse_f,loss_tot_v,lr_net_v,lr_finger_v,ep,sep=' ',end='\n',file=fileOU)
       print("Testing model at global step",accumul," and epoch ",ep," val_lossE ",rmse_e," val_lossF ",rmse_f," loss_Tot ",loss_tot_v," lr_net ",lr_net_v," lr_finger ",lr_finger_v,sep=' ',end='\n')
       metrics_log.log(
           global_step=int(accumul),
           epoch=int(ep),
           rmse_e=rmse_e,
           rmse_f=rmse_f,
           loss_tot=loss_tot_v,
           lr_net=lr_net_v,
           lr_finger=lr_finger_v,
       )
       print("We are at epoch ",ep)
       fileOU.flush()
       out_time.flush()
