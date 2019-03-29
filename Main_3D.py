#   Ising Model in Python.

#   28-03-2019.
#   Written by Anand Mahesh.
#   Python 3.7.
#   NumPy has been installed and used in this project.
#   Numba has been installed and used in this project.
#   Tools used: Visual Studio Code, GitHub Desktop.

from numba import jit

import random
import numpy
import time
import math
import csv


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
stringreader=""         #   variable to read files from text to be later converted to int
iterator=0              #   to be used with for loop / dummy operation
iterator2=0             #   to be used  for loop / dummy operations

print("\n")
print("MONTE CARLO QUASI 3D ISING MODEL\n")
print("Monte Carlo Statistics for Quasi 3D Ising Model with periodic boundary conditions\n")
print("The critical temperature is approximately 2.3, as seen on Chandler p. 123.\n")

ising = open("ising.in", "r")       #   This section is for reading input parameters and assigning it to global variables
                                       
next(ising)
stringreader=(ising.readline())
nrows=int(stringreader)

next(ising)
stringreader=(ising.readline())
ncols=int(stringreader)

next(ising)
stringreader=(ising.readline())
nlayers=int(stringreader)

next(ising)
stringreader=(ising.readline())
npass=int(stringreader)

next(ising)
stringreader=(ising.readline())
nequil=int(stringreader)

next(ising)
stringreader=(ising.readline())
high_temp=float(stringreader)

next(ising)
stringreader=(ising.readline())
low_temp=float(stringreader)

next(ising)
stringreader=(ising.readline())
temp_interval=float(stringreader)

next(ising)
stringreader=(ising.readline())
ConfigType=int(stringreader)

ising.close()



# End of input parameter reader section

iterator = nrows
iterator2 = ncols

if(nrows%2!=0):
    iterator+=1
if(ncols%2!=0):
    iterator2+=1

print("Running program for %d rows and %d columns\n" % (iterator,iterator2))

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


# def magnetization_calc(output_count,magnetization,magnetization_ave,magnetization2_ave,energy,nlayers,iterator,iterator2):
#             output_count+=1
#             magnetization = numpy.sum(a[0:nlayers,1:iterator-2,1:iterator-2])/(nlayers*iterator*iterator2*1.0)
#             magnetization_ave = magnetization_ave + magnetization
#             magnetization2_ave = magnetization2_ave + magnetization**2
#             energy = 0.00

#             return output_count,magnetization,magnetization_ave,magnetization2_ave,energy


spin_attribute = open("spin_array_attribute.csv", "w")
spin_attribute.write("number of rows        :"+str(nrows))
spin_attribute.write("\nnumber of columns   :"+str(ncols))
spin_attribute.write("\nnumber of layers    :"+str(nlayers))


nscans=int((high_temp-low_temp)/temp_interval+1)        #   Determining the number of scans

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
    
    beta  =  1.0/temp
    output_count   =   0
    energy_ave  =  0.0
    energy2_ave  =  0.0
    magnetization_ave  =  0.0
    magnetization2_ave  =  0.0
    
    a=start_matrix

    #   Main loop containing Monte Carlo algorithm

    for ipass in range(0,npass+1):
        
        if(ipass>nequil):
           
            output_count+=1
            magnetization = numpy.sum(a[0:nlayers,1:iterator-2,1:iterator-2])/(nlayers*iterator*iterator2*1.0)
            magnetization_ave = magnetization_ave + magnetization
            magnetization2_ave = magnetization2_ave + magnetization**2
            energy = 0.00

            for k in range(0,nlayers):              #   Depth
                for i in range(0,iterator):         #   Row
                    for j in range(0,iterator2):    #   Column
                   
                        if(d!=0 or d!=nlayers-1):   #   When the matrix element is not on the top or bottom layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n]+a[d-1,m,n])
                        elif(d==0):                 #   When the matrix element is on the bottom layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n])
                        else:                       #   When the matrix element is on the top layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d-1,m,n])


            energy = energy / (nlayers*iterator*iterator2*2.0)
            energy_ave = energy_ave + energy
            energy2_ave = energy2_ave + energy**2

        ran0=pick_random(ran0) 
        m=int((iterator-2)*ran0)  
        ran0=pick_random(ran0)
        n=int((iterator2-2)*ran0)
        ran0=pick_random(ran0)
        d=int((nlayers-1)*ran0)
        trial_spin=-1*(a[d,m,n]) 

        if(d!=0 or d!=nlayers-1):   #   When the matrix element is not on the top or bottom layer
            DeltaU = -1*(trial_spin*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n]+a[d-1,m,n])*2)
        elif(d==0):                 #   When the matrix element is on the bottom layer
            DeltaU = -1*(trial_spin*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n])*2)
        else:                       #   When the matrix element is on the top layer
            DeltaU = -1*(trial_spin*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d-1,m,n])*2)
        
        ran0=pick_random(ran0)
        log_eta=math.log(ran0+(1e-10))
        
        if(-beta*DeltaU>log_eta):
            
            a[d,m,n]=trial_spin

            if(m==1):
                a[d,iterator-1,n]=trial_spin
            
            if(m==iterator-1):
                a[d,0,n]=trial_spin
            
            if(n==1):
                a[d,m,iterator2-1]=trial_spin
            
            if(n==iterator2-1):                
                a[d,m,0]=trial_spin
    
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

print("\nProgram Completed\n")

spin.close()                                            #   Closing open files.This part is important as open files may not allow writing of new data
magnet.close()
energyObj.close()

Profiler = open("Program_Profile.csv","a+")
time_elapsed=(time.perf_counter()-time_start)           #   Program execuion time profiler
Profiler.write("\n"+str(time_elapsed)+"")
Profiler.close()

#   THE END