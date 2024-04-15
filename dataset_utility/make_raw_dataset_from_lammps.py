import numpy as np
import sys

traj_name="traj.xyz"
trajf_name="trajf.xyz"
thermo_name="thermo.out"

enepot=np.loadtxt("thermo.out")[:,[0,5]]
nf=int(enepot.shape[0]/10)


traj_file=open(traj_name,'r')
trajf_file=open(trajf_name,'r')


N=3072
NO=1024
NH=2048

pos_arr=np.zeros((nf,N,3))
force_arr=np.zeros((nf,N,3))
box_arr=np.zeros((nf,6))
ene_arr=np.zeros((nf,1))


for fr in range(nf):

    
    line1=traj_file.readline()
    line2=trajf_file.readline()

    t1=int(traj_file.readline()) 
    t2=int(trajf_file.readline())

    indexene=enepot[:,0]==t1
    ene_arr[fr]=enepot[indexene,1]
   

    line1=traj_file.readline()
    line2=trajf_file.readline()

    N1=int(traj_file.readline())
    N2=int(trajf_file.readline())

    line1=traj_file.readline()
    line2=trajf_file.readline()

    box1=traj_file.readline().split()
    box2=trajf_file.readline().split()

    bsx=-float(box1[0])

    box_arr[fr,0]=float(box1[1])+bsx

    box1=traj_file.readline().split()
    box2=trajf_file.readline().split()

    bsy=-float(box1[0])

    box_arr[fr,3]=float(box1[1])+bsy

    box1=traj_file.readline().split()
    box2=trajf_file.readline().split()

    bsz=-float(box1[0])

    box_arr[fr,5]=float(box1[1])+bsz


    line1=traj_file.readline()
    line2=trajf_file.readline()

    index0=0
    indexH=NO
    for par in range(N):
        pos_line=traj_file.readline().split()
        force_line=trajf_file.readline().split()
        if int(pos_line[1])==1:
             pos_arr[fr,index0,:]=[float(pos_line[-3])+bsx,float(pos_line[-2])+bsy,float(pos_line[-1])+bsz]
             force_arr[fr,index0,:]=[float(force_line[-3]),float(force_line[-2]),float(force_line[-1])]
             index0=index0+1
        elif int(pos_line[1])==2:
             pos_arr[fr,indexH,:]=[float(pos_line[-3])+bsx,float(pos_line[-2])+bsy,float(pos_line[-1])+bsz]
             force_arr[fr,indexH,:]=[float(force_line[-3]),float(force_line[-2]),float(force_line[-1])]
             indexH=indexH+1
        else: 
             sys.exit("Error type is not 1 or 2 "+str(pos_line[1]))

np.savetxt("pos_raw.dat",pos_arr.reshape((nf,-1)))
np.savetxt("force_raw.dat",force_arr.reshape((nf,-1)))
np.savetxt("box_raw.dat",box_arr.reshape((nf,-1)))
np.savetxt("ene_raw.dat",ene_arr.reshape((nf,-1)))




