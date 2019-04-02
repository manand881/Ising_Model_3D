from datetime import date, datetime
import time
import os

def Output_Path_Set():                                          #   Function to create folder based on date and time of execution and set it as working directory 
    now = datetime.now()                                        #   Fetching current date and time
    path_current_time = now.strftime("%H-%M-%S__%d-%b-%Y")      #   Formatting date and time to suit windows nomenclature

    try:                                                        #   Try catch to open file name based on path_current_time
        os.makedirs(path_current_time)
        os.chdir(path_current_time)    
    except FileExistsError:                                     #   Display error in case of file already existing
        print(path_current_time,"FIle already exists")

    return path_current_time                                    #   Returning date time to main program