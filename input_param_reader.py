def Ising_input():

    nrows = 20              #  number of rows of spins (even number)
    
    ncols = 20              #  number of columns of spins (even number)

    nlayers = 3             #  number of layers in the quasi 3D Matrix

    npass = 60000           #  number of passes for each temperature

    nequil = 40000          #  number of equilibration steps for each temperature

    high_temp = 5.0         #  temperature to start scan at

    low_temp = 0.92         #  temperature to finish scan at

    temp_interval = 0.01    #  scanning interval

    ConfigType = 1          #  1: checkerboard, 2: interface, 3: unequal interface, 4: Random Matrix

    return nrows, ncols, nlayers, npass, nequil, high_temp, low_temp, temp_interval, ConfigType
