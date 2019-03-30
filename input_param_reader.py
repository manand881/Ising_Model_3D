def ising_input():

    nrows = 20              #  number of rows of spins (even number)
    
    ncols = 20              #  number of columns of spins (even number)

    npass = 15500           #  number of passes for each temperature

    nlayers = 3             #  number of layers in the quasi 3D Matrix

    nequil = 13000          #  number of equilibration steps for each temperature

    high_temp = 4.0         #  temperature to start scan at

    low_temp = 0.92         #  temperature to finish scan at

    temp_interval = 0.01    #  scanning interval

    ConfigType = 1          #  1: checkerboard, 2: interface, 3: unequal interface, 4: Random Matrix

    return nrows, ncols, nlayers, npass, nequil, high_temp, low_temp, temp_interval, ConfigType
