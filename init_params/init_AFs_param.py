import numpy as np
from numpy.random import seed
from numpy import random
from numpy.random import default_rng



def filtered_matrix(map_to_build):
    nrow=len(map_to_build)
    ncol=sum(map_to_build)
    matrix=np.zeros((nrow,ncol),dtype='float32')
    prev=0
    for k in range(nrow):
        matrix[k,prev:prev+map_to_build[k]]=1.
        prev+=map_to_build[k]
    return matrix

def gen_map_type_AFs(full_param):
    map_rad_afs=full_param['map_rad_afs']
    map_ang_afs=full_param['map_ang_afs']
    nt=len(map_rad_afs)
    nt_couple=int(nt*(nt+1)/2)
    tot_rad_afs=[sum(map_rad_afs[k]) for k in range(nt)]
    tot_ang_afs=[sum(map_ang_afs[k]) for k in range(nt)]
    type_emb2b=[filtered_matrix(map_rad_afs[tt]) for tt in range(nt)]
    type_emb3b=[filtered_matrix(map_ang_afs[tt]) for tt in range(nt)]
    type_emb=[[type_emb2b[k],type_emb3b[k]] for k in range(nt)]
    return type_emb

###Initialize alphas
def init_AFs_param(restart,full_param,nt,seed_par):
    np.random.set_state(seed_par)
    nt_couple=int(nt*(nt+1)/2)
    try:
        alpha_bound=float(full_param['alpha_bound'])
        print("alpha_nes: alphas will be upper-bound to custom",alpha_bound,sep=' ',end='\n')
    except:
        alpha_bound=1.
        print("alpha_nes: alphas will be upper-bound to default",alpha_bound,sep=' ',end='\n')
    limit=alpha_bound
    limit3b=alpha_bound
    if restart=='no':
        nalpha_r_list=full_param['Total_number_radial_AFs'].split()
        nalpha_a_list=full_param['Total_number_angular_AFs'].split()

        nalpha_r_arr=np.array([int(k) for k in nalpha_r_list]).reshape((nt,2))
        nalpha_a_arr=np.array([int(k) for k in nalpha_a_list]).reshape((nt,2))
        ###Initialize radial AFS parameters
        init_alpha2b=[(np.random.rand((nalpha_r_arr[k,1]*nt))*2*limit-limit).reshape((nt,nalpha_r_arr[k,1])).astype('float32') for k in range(nt)]
        ###Initialize angular AFS parameters
        init_alpha3b=[]
        for k in range(nt):
            vec=np.zeros((nalpha_a_arr[k,1]*nt_couple,3),dtype='float32')
            vec[:,:2]=(np.random.rand((nalpha_a_arr[k,1]*nt_couple*2))*2*limit3b-limit3b).reshape((nalpha_a_arr[k,1]*nt_couple,2)).astype('float32')
            vec[:,2]=(np.random.rand((nalpha_a_arr[k,1]*nt_couple))*-10).reshape(nalpha_a_arr[k,1]*nt_couple).astype('float32')
            init_alpha3b.append(vec.reshape((nt_couple,nalpha_a_arr[k,1]*3)))
        #Initialised Z for each AFS
        init_mu=[(np.random.rand(nalpha_r_arr[k,1]+nalpha_a_arr[k,1])*2*limit-limit).astype('float32')
                for k in range(nt)]
        ###Initialize Ck parameters (only for mixtures)
        if nt>1:
           initial_type_emb=gen_map_type_AFs(full_param)
        else:
           initial_type_emb_2b=[(np.ones(nt*nalpha_r_arr[k,1])).reshape((nt,nalpha_r_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb_3b=[(np.ones(nt_couple*nalpha_a_arr[k,1])).reshape((nt_couple,nalpha_a_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(nt)]
    ##Initialise only afs by reading them from file. State of optimizer is started from scratch.
    elif restart=='only_afs' or restart=='all_params':
         afs_param=full_param['afs_param_folder']
         init_mu=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_mu.dat',dtype='float32') for k in range(nt)]
         init_alpha2b=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_2body.dat',dtype='float32') for k in range(nt)]
         init_alpha3b=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_3body.dat',dtype='float32') for k in range(nt)]
         nalpha_r_arr=np.array([[k,init_alpha2b[k].shape[1]] for k in range(nt)])
         nalpha_a_arr=np.array([[k,int(init_alpha3b[k].shape[1]/3)] for k in range(nt)])
         if nt>1:
             initial_type_emb_2b=[np.loadtxt(afs_param+'/type'+str(k)+'_type_emb_2b.dat',dtype='float32') for k in range(nt)]
             initial_type_emb_3b=[np.loadtxt(afs_param+'/type'+str(k)+'_type_emb_3b.dat',dtype='float32') for k in range(nt)]
             initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(nt)]
         else:
           initial_type_emb_2b=[(np.ones(nt*nalpha_r_arr[k,1])).reshape((nt,nalpha_r_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb_3b=[(np.ones(nt_couple*nalpha_a_arr[k,1])).reshape((nt_couple,nalpha_a_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(nt)]
    else:
        init_alpha2b=[np.loadtxt(restart+'/type'+str(k)+'_alpha_2body.dat',dtype='float32') for k in range(nt)]
        init_alpha3b=[np.loadtxt(restart+'/type'+str(k)+'_alpha_3body.dat',dtype='float32') for k in range(nt)]
        nalpha_r_arr=np.array([[k,init_alpha2b[k].shape[1]] for k in range(nt)])
        nalpha_a_arr=np.array([[k,int(init_alpha3b[k].shape[1]/3)] for k in range(nt)])
        init_mu=[np.loadtxt(restart+'/type'+str(k)+'_alpha_mu.dat',dtype='float32') for k in range(nt)]
        if nt>1:
            initial_type_emb_2b=[np.loadtxt(restart+'/type'+str(k)+'_type_emb_2b.dat',dtype='float32') for k in range(nt)]
            initial_type_emb_3b=[np.loadtxt(restart+'/type'+str(k)+'_type_emb_3b.dat',dtype='float32') for k in range(nt)]
            initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(nt)]
        else:
           initial_type_emb_2b=[(np.ones(nt*nalpha_r_arr[k,1])).reshape((nt,nalpha_r_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb_3b=[(np.ones(nt_couple*nalpha_a_arr[k,1])).reshape((nt_couple,nalpha_a_arr[k,1])).astype('float32') for k in range(nt)]
           initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(nt)]
    print("alpha_nes: Two-Body Atomic fingerprints are ",end='\n')
    print("alpha_nes: atom type   number\n")
    for k in range(nt):
        print("alpha_nes:     ",nalpha_r_arr[k,0],"        ",nalpha_r_arr[k,1])
        print("alpha_nes: Three-Body Atomic fingerprints are ",end='\n')
        print("alpha_nes: atom type   number\n")
    for k in range(nt):
        print("alpha_nes:      ",nalpha_a_arr[k,0],"        ",nalpha_a_arr[k,1])

    return init_alpha2b,init_alpha3b,init_mu,initial_type_emb,np.random.get_state()
