from numba import jit

import random
import numpy
import math

#   Function to generate uniform random numbers

@jit(nopython=True)
def pick_random(ran0):
    
    ran0=round(random.uniform(0,1),12)
    
    return ran0 

#   End of function

def magnetization_sum(nlayers,iterator,iterator2,start_matrix,magnetization_sumer):
    magnetization_sumer=0
    for k in range(0,nlayers):                      #   Depth
        for i in range(1,iterator):                 #   Rows
            for j in range(1,iterator2):            #   Columns
                magnetization+=start_matrix[k,i,j]
    print(magnetization_sumer)    

#   Function to perfrom Montecarlo loop

@jit(nopython=True)
def Monte_Carlo(m , n , d , i , j , k , ipass , npass , nequil , iterator , iterator2 , nlayers , ran0 , a , magnetization , magnetization_ave , magnetization2_ave , energy , beta , DeltaU , output_count,energy_ave,energy2_ave ):
    
    for ipass in range(0,npass+1): 
    
        if(ipass>nequil):
            
            output_count+=1            
            magnetization = numpy.sum(a[0:nlayers,1:iterator-1,1:iterator-1])/(nlayers*iterator*iterator2*1.0)     #   Calling magnetization summing function 
            magnetization_ave = magnetization_ave + magnetization
            magnetization2_ave = magnetization2_ave + magnetization**2
            energy = 0.00

            for k in range(0,nlayers):              #   Depth
                for i in range(0,iterator):         #   Row
                    for j in range(0,iterator2):    #   Column

                        if(d!=0 and d!=(nlayers-1)):    #   When the matrix element is not on the top or bottom layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n]+a[d-1,m,n])
                        elif(d==0):                     #   When the matrix element is on the bottom layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n])
                        else:                           #   When the matrix element is on the top layer
                            energy = energy - a[d,m,n]*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d-1,m,n])


            energy = energy / (nlayers*iterator*iterator2*2.0)
            energy_ave = energy_ave + energy
            energy2_ave = energy2_ave + energy**2

        ran0=pick_random(ran0) 
        m=int((iterator-2)*ran0)                    #   Picking random spin row number
        ran0=pick_random(ran0)
        n=int((iterator2-2)*ran0)                   #   Picking random spin column number
        ran0=pick_random(ran0)
        d=int((nlayers)*ran0)                       #   Picking random spin depth number
        trial_spin=-1*(a[d,m,n]) 

        if(d!=0 and d!=(nlayers-1)):    #   When the matrix element is not on the top or bottom layer
            DeltaU = -1*(trial_spin*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n]+a[d-1,m,n])*2)
        elif(d==0):                     #   When the matrix element is on the bottom layer
            DeltaU = -1*(trial_spin*(a[d,m-1,n]+a[d,m+1,n]+a[d,m,n-1]+a[d,m,n+1]+a[d+1,m,n])*2)
        else:                           #   When the matrix element is on the top layer
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

    return m , n , d , i , j , k , ipass , npass , nequil , iterator , iterator2 , nlayers , ran0 , a , magnetization , magnetization_ave , magnetization2_ave , energy , beta , DeltaU , output_count,energy_ave,energy2_ave

#   End of function
