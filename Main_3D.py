#   Ising Model in Python.

#   28-03-2019.
#   Written by Anand Mahesh.
#   Python 3.7.
#   NumPy has been installed and used in this project.
#   Numba has been installed and used in this project.
#   Tools used: Visual Studio Code, GitHub Desktop.

from Input_param_reader     import Ising_input      #   Python Function in the same directory as the Main.py File
from Montecarlo             import Monte_Carlo      #   Python Function in the same directory as the Main.py File
from numba                  import jit              #   Python Package to be downloaded manually 
from Path                   import Output_Path_Set  #   Python Function to create output folder by date and time and set it as working directory

import random
import numpy
import time
import math
import csv
import os



time_start = time.perf_counter()    #   For Program Runtime Profiling. Time.clock() has been depreciated 



i=0                     #   Dummy Integer
j=0                     #   Dummy Integer
k=0                     #   Dummy Integer
m=0                     #   Dummy Integer
n=0                     #   Dummy Integer
d=0                     #   Dummy Integer
nrows=0                 #   Number of Rows in A
ncols=0                 #   Number of Columns in A
nlayers=0               #   Number of Layers in Quasi 3D Matrix 
temp=0                  #   Temperature
beta=0                  #   Inverse Temperature
ConfigType=0            #   Starting Configuration type
npass=0                 #   number of passes for MC algorithm
ipass=0                 #   the current pass number
nequil=0                #   number of equilibration steps
trial_spin=0            #   values of changed spin
high_temp=0             #   starting temp for scan
low_temp=0              #   final temp for scan
temp_interval=0         #   interval between scan points
nscans=0                #   number of scans (each at diff T)
iscan=1                 #   current number
iscan1=0                #   current number
DeltaU=0                #   change in energy between 2 configs
log_eta=0               #   log of random number to compare to
magnetization=0         #   magnetization of all spins in lattice
magnetization_ave=0     #   cumulative average magnetization
magnetization2_ave=0    #   cumulative average of mag. squared
energy=0                #   energy of all spins in lattice
energy_ave=0            #   cumulative average of energy
energy2_ave=0           #   cumulative average of energy squared
output_count=0          #   Number of times things have been added to averages
ran0=0                  #   T B C
iterator=0              #   to be used with for loop / dummy operation
iterator2=0             #   to be used  for loop / dummy operations

print("\n")
print("MONTE CARLO QUASI 3D ISING MODEL\n")
print("Monte Carlo Statistics for Quasi 3D Ising Model with periodic boundary conditions\n")
print("The critical temperature is approximately 2.3, as seen on Chandler p. 123.\n")

#   This section is for reading input parameters and assigning it to global variables

nrows, ncols, nlayers, npass, nequil, high_temp, low_temp, temp_interval, ConfigType=Ising_input()

#   End of input parameter reader section

iterator = nrows        #   Setting iterator to be used as number of rows value
iterator2 = ncols       #   Setting iterator to be used as number of columns value

if(nrows%2!=0):
    iterator+=1
if(ncols%2!=0):
    iterator2+=1

print("Running program for %d rows, %d columns and %d layers\n" % (iterator,iterator2,nlayers))

#   Matrix arrays are stored as a[depth,row,column] manner in Numpy



a=numpy.ones((nlayers,iterator,iterator2),dtype=int)
start_matrix=numpy.ones((nlayers,iterator,iterator2),dtype=int)



#   Functions



#   Function to generate uniform random numbers

@jit(parallel=True)
def pick_random(ran0):
    
    ran0=round(random.uniform(0,1),12)
    
    return ran0 

#   End of function



#   Function to obtain magnetization value

@jit(nopython=True)
def magnetization_sum(nlayers,iterator,iterator2,a):
    return numpy.sum(a[0:nlayers,1:iterator-1,1:iterator-1])/(nlayers*iterator*iterator2*1.0)

#   End of function



path=Output_Path_Set()


input_config=open("Input_Config.csv","w+")                                #   To write input configuration to output folder in a seperate file for future use.
input_config.write("Number of Rows          :"+str(nrows))              
input_config.write("\nNumber of Columns       :"+str(ncols))
input_config.write("\nValue of npass          :"+str(npass))
input_config.write("\nValue of nequil         :"+str(nequil))
input_config.write("\nValue of high_temp      :"+str(high_temp))
input_config.write("\nValue of low_temp       :"+str(low_temp))
input_config.write("\nValue of temp_interval  :"+str(temp_interval))
input_config.write("\nConfigType              :"+str(ConfigType))
input_config.close()


spin_attribute = open("spin_array_attribute.csv", "w")
spin_attribute.write("number of rows        :"+str(nrows))
spin_attribute.write("\nnumber of columns   :"+str(ncols))
spin_attribute.write("\nnumber of layers    :"+str(nlayers))

nscans=int((high_temp-low_temp)/temp_interval+1)            #   Determining the number of scans

spin_attribute.write("\nnumber of scans     :"+str(nscans))
spin_attribute.write("\n2")

spin_attribute.close()

spin = open("spin_array.csv","w+")
spin_writer=csv.writer(spin)
spin_row=["temp","i","j","k","a[i,j]"]
spin_writer.writerow(spin_row)

magnet = open("magnetization.csv","w+")
magnet.write("Temp , Ave_magnetization , Ave_magnetization^2 , Susceptibility")
magnet.write("\n")
magnet_writer=csv.writer(magnet)

energyObj = open("energy.csv","w+")
energyObj.write("Temp , Ave_energy , Ave_energy^2 , C_v")
energyObj.write("\n")
energy_writer=csv.writer(energyObj)



#   Section for choosing Configtype



if(ConfigType==1):                                              
        
    #   Checkerboard Pattern Matrix
                
    start_matrix[1::2,::2,::2] = -1             #   Depth
    start_matrix[::2,1::2,::2] = -1             #   Row
    start_matrix[::2,::2,1::2] = -1             #   Column


elif(ConfigType==2):
        
    #   Interface Pattern Matrix

    for k in range(0,nlayers):                  #   Depth
        for i in range(0,iterator):             #   Row
            for j in range(0,iterator2):        #   Column
                if(j>=iterator2/2):
                    dummyval=-1
                else:
                    dummyval=1
                start_matrix[:,:,j]=dummyval
    dummyval=0   

elif(ConfigType==3):

    #   Unequal Interface Pattern Matrix

    for k in range(0,nlayers):                  #   Depth
        for i in range(0,iterator):             #   Row
            for j in range(0,iterator2):        #   Column
                if(j>=iterator2/4):
                    dummyval=-1
                else:
                    dummyval=1
                start_matrix[:,:,j]=dummyval
    dummyval=0

elif(ConfigType==4):

#   Random Pattern Matrix
    
    for k in range(0,nlayers):                  #   Depth
        for i in range(0,iterator):             #   Row
            for j in range(0,iterator2):        #   Column
                dummy=pick_random(ran0)
                if(dummy>=0.5):
                    dummy=1
                else:
                    dummy=-1
                start_matrix[k,i,j]=dummy

else:
    print("Error! Check ConfigType parameter in ising.in")



#   Scan Loop



for iscan in range(1,nscans+1):                                         #   Main for loop    
    temp = float(round((high_temp - temp_interval*(iscan-1)), 3))       #   rounding off to two decimal places for optimisation purposes 
    print("Running Program for Temperature : "+str(temp)+"\n")
    
    beta  =  1.0/temp                           #   Reseting variables to initial values
    output_count   =   0
    energy_ave  =  0.0
    energy2_ave  =  0.0
    magnetization_ave  =  0.0
    magnetization2_ave  =  0.0
    
    a=start_matrix                              #   Reseting matrix a to initial congiguration

    #   Main loop containing Monte Carlo algorithm
    
    m , n , d , i , j , k , ipass , npass , nequil , iterator , iterator2 , nlayers , ran0 , a , magnetization , magnetization_ave , magnetization2_ave , energy , beta , DeltaU , output_count , energy_ave , energy2_ave = Monte_Carlo( m , n , d , i , j , k , ipass , npass , nequil , iterator , iterator2 , nlayers , ran0 , a , magnetization , magnetization_ave , magnetization2_ave , energy , beta , DeltaU , output_count,energy_ave,energy2_ave )
    
    #   End Monte carlo pases



    for k in range(0,nlayers):                      #   Depth
        for i in range(0,iterator):                 #   Rows
            for j in range(0,iterator2):            #   Columns
                spin_row=[temp,k,i,j,a[k,i,j]]
                spin_writer.writerow(spin_row)
    
    magnet_row=[temp , abs(magnetization_ave/output_count) , magnetization2_ave/output_count , beta*(magnetization2_ave/output_count - (magnetization_ave/output_count)**2)]
    magnet_writer.writerow(magnet_row)
    
    energy_row=[temp , energy_ave/output_count , energy2_ave/output_count , (beta**2)*(energy2_ave/output_count - (energy_ave/output_count)**2)]
    energy_writer.writerow(energy_row)

#   End Scan Loop

print("\nProgram completed.\n\nOpen folder",path,"to view output.\n\n")

spin.close()                                         #   Closing open files.This part is important as open files may not allow writing of new data
magnet.close()
energyObj.close()

Profiler = open("Program_Profile.csv","a+")
time_elapsed=(time.perf_counter()-time_start)        #   Program execuion time profiler
time_elapsed=round(time_elapsed,5)
Profiler.write("\nProgram FInished running in "+str(time_elapsed)+" Seconds on "+str(time.ctime()))
Profiler.close()

#   THE END