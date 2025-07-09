import numpy as np
from numpy.random import seed
from numpy import random
from numpy.random import default_rng

import sys



def filtered_matrix(map_to_build):
    nrow=len(map_to_build)
    ncol=sum(map_to_build)
    matrix=np.zeros((nrow,ncol),dtype='float64')
    prev=0
    for k in range(nrow):
        matrix[k,prev:prev+map_to_build[k]]=1.
        prev+=map_to_build[k]
    return matrix

def gen_map_type_AFs(full_param):
    map_rad_afs=full_param['map_rad_afs']
    map_ang_afs=full_param['map_ang_afs']
    number_of_NN=len(map_rad_afs)
    #nt_couple=int(nt*(nt+1)/2)
    tot_rad_afs=[[k,sum(map_rad_afs[k])] for k in range(number_of_NN)]
    tot_ang_afs=[[k,sum(map_ang_afs[k])] for k in range(number_of_NN)]

    tot_rad_afs_arr=np.array(tot_rad_afs).reshape((number_of_NN,2))
    tot_ang_afs_arr=np.array(tot_ang_afs).reshape((number_of_NN,2))
    type_emb2b=[filtered_matrix(map_rad_afs[tt]) for tt in range(number_of_NN)]
    type_emb3b=[filtered_matrix(map_ang_afs[tt]) for tt in range(number_of_NN)]
    type_emb=[[type_emb2b[k],type_emb3b[k]] for k in range(number_of_NN)]
    return tot_rad_afs_arr,tot_ang_afs_arr,type_emb,number_of_NN


###Initialize alphas
def init_AFs_param(restart,full_param,number_of_interaction,seed_par):
    np.random.set_state(seed_par)
    #nt_couple=int(nt*(nt+1)/2)
    map_rad_afs=full_param['map_rad_afs']
    for el in map_rad_afs.keys():
        if (len(map_rad_afs[el])!=number_of_interaction):
            print("Alpha_nes: The number of interaction is ",number_of_interaction, "but you indicate ",len(map_rad_afs[el])," number in the map_rad_afs section!")
            sys.exit()
    map_ang_afs=full_param['map_ang_afs']
    nt_couple_interaction=int((number_of_interaction+1)*number_of_interaction/2)
    for el in map_ang_afs.keys():
        if (len(map_ang_afs[el])!=nt_couple_interaction):
            print("Alpha_nes: The number of possible angular interaction is ",nt_couple_interaction, "but you indicate ",len(map_ang_afs[el])," number in the map_ang_afs section!")
            sys.exit()    
    try:
        alpha_bound=float(full_param['alpha_bound'])
        print("alpha_nes: alphas will be upper-bound to custom",alpha_bound,sep=' ',end='\n')
    except:
        alpha_bound=1.
        print("alpha_nes: alphas will be upper-bound to default",alpha_bound,sep=' ',end='\n')
    limit=alpha_bound
    limit3b=alpha_bound
    if full_param['restart']=='no':
        [nalpha_r_arr,nalpha_a_arr,initial_type_emb,number_of_NN]=gen_map_type_AFs(full_param)
        
        ###Initialize radial AFS parameters
        init_alpha2b=[]
        for k in range(number_of_NN):
            jump_xx=0
            vec2b=np.zeros((number_of_interaction,nalpha_r_arr[k,1]))
            alphas=(np.random.rand(nalpha_r_arr[k,1])*2*limit-limit).astype('float64')
            for xx,param in enumerate(map_rad_afs[k]):
                vec2b[xx,jump_xx:jump_xx+param]=alphas[jump_xx:jump_xx+param]
                jump_xx=jump_xx+param
            init_alpha2b.append(np.array(vec2b).reshape((number_of_interaction,nalpha_r_arr[k,1])))
        ###Initialize angular AFS parameters
        init_alpha3b=[]
        for k in range(number_of_NN):
            vec=np.zeros((nt_couple_interaction,nalpha_a_arr[k,1],3),dtype='float64')
            delta_gamma=(np.random.rand((nalpha_a_arr[k,1]*2))*2*limit3b-limit3b).astype('float64')
            beta=(np.random.rand(nalpha_a_arr[k,1])*-10).astype('float64')
            jump_xx=0
            for xx,param in enumerate(map_ang_afs[k]):
                vec[xx,jump_xx:jump_xx+param,:2]=delta_gamma[jump_xx*2:(jump_xx+param)*2].reshape((-1,2))
                vec[xx,jump_xx:jump_xx+param,2]=beta[jump_xx:jump_xx+param]
                jump_xx=jump_xx+param
            init_alpha3b.append(np.array(vec).reshape((nt_couple_interaction,nalpha_a_arr[k,1]*3)))
        #Initialised Z for each AFS
        init_mu=[(np.random.rand(nalpha_r_arr[k,1]+nalpha_a_arr[k,1])*2*limit-limit).astype('float64')
                for k in range(number_of_NN)]
    ##Initialise only afs by reading them from file. State of optimizer is started from scratch.
    else :
        map_rad_afs=full_param['map_rad_afs']
        number_of_NN=len(map_rad_afs)
        if full_param['restart']==['only_afs','full_param']:
           afs_param=full_param['params_folder']
        else:
           afs_param=restart
        print("alpha_nes: AF parameters load from folder ",afs_param) 
        init_mu=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_mu.dat',dtype='float64') for k in range(number_of_NN)]
        init_alpha2b=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_2body.dat',dtype='float64').reshape((number_of_interaction,-1)) for k in range(number_of_NN)]
        init_alpha3b=[np.loadtxt(afs_param+'/type'+str(k)+'_alpha_3body.dat',dtype='float64').reshape((nt_couple_interaction,-1)) for k in range(number_of_NN)]
        nalpha_r_arr=np.array([[k,init_alpha2b[k].shape[1]] for k in range(number_of_NN)])
        nalpha_a_arr=np.array([[k,int(init_alpha3b[k].shape[1]/3)] for k in range(number_of_NN)])
        initial_type_emb_2b=[np.loadtxt(afs_param+'/type'+str(k)+'_type_emb_2b.dat',dtype='float64') for k in range(number_of_NN)]
        initial_type_emb_3b=[np.loadtxt(afs_param+'/type'+str(k)+'_type_emb_3b.dat',dtype='float64') for k in range(number_of_NN)]
        initial_type_emb=[[initial_type_emb_2b[k],initial_type_emb_3b[k]] for k in range(number_of_NN)]
    print("alpha_nes: Two-Body Atomic fingerprints are ",end='\n')
    print("alpha_nes: NN Index   number\n")
    for k in range(number_of_NN):
        print("alpha_nes:     ",nalpha_r_arr[k,0],"        ",nalpha_r_arr[k,1])
    print("alpha_nes: Three-Body Atomic fingerprints are ",end='\n')
    print("alpha_nes: NN index   number\n")
    for k in range(number_of_NN):
        print("alpha_nes:      ",nalpha_a_arr[k,0],"        ",nalpha_a_arr[k,1])

    return init_alpha2b,init_alpha3b,init_mu,initial_type_emb,np.random.get_state()
breakpoint()
