# Python Program for Simulating Quasi 3D Ising Model Using Multi Processor

This program was written in march 2019 as an internship assignment while I was in NIT-Manipur as an Intern under [Dr.Shagolsem Lenin Singh](http://nitmanipur.ac.in/emp_profile_New.aspx?nDeptID=iaeke).

This program tries to simulate the Quasi 3D Ising Model in python. This is just a Trial.

## Installation

This Program needs Numba And Numpy to run as it uses them as dependencies.

This program only runs on python3 and not on python2. ensure that you are runing python3(Preferably a 64 bit version for better results).

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Numba](https://pypi.org/project/numba/) and [Numpy](https://pypi.org/project/numpy/).

```bash
pip install numba
pip install numpy
```

## Usage

```python
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
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. I would be happy to clarify any doubts by [mail](manand881@gmail.com).

Please make sure to update tests as appropriate.

## Gratitude

I would like to state that I am deeply grateful to :- 

* My Professor [Dr.Shagolsem Lenin Singh](http://nitmanipur.ac.in/emp_profile_New.aspx?nDeptID=iaeke) for the oppurtunity to work under him and for all the guidance and support that he has given me.

* My Professors from [MES Degree College](https://mesinstitutions.in/web/mes-degree-college) for all their support and encouragement withouth which i would not be where I am today.

* Friends and Family for bearing with me and being supportive for all my endeavour.


## License
[MIT](https://choosealicense.com/licenses/mit/)
