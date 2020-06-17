# Importing the toolbox (takes several seconds)
import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import deeplabcut

# The plotting functions below are put here for simplicity and so that the user can edit them. Note that they
# (or variants thereof) are in fact in standard DLC and accessible via:



# this is example data from the public project: https://github.com/AlexEMG/DeepLabCut/tree/master/examples/openfield-Pranav-2018-10-30
video='Vglut-cre C137 F3-_2'
DLCscorer='DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500'
# Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.h5
dataname = str(Path(video).stem) + DLCscorer + '.h5'

#loading output of DLC
Dataframe = pd.read_hdf(os.path.join(dataname))


#Let's have a look at the data:

#these structures are awesome to manipulate, how -->> see pandas https://pandas.pydata.org/pandas-docs/stable/index.html
print(Dataframe.head())