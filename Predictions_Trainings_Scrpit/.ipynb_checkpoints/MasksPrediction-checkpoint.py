#!/usr/bin/env python
# coding: utf-8


#from __future__ import print_function, unicode_literals, absolute_import, division


# In[0]:

#import os
#import time

#TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/Flo4'
#TimeCount = 0
#TimeThreshold = 3600*36
#while os.path.exists(TriggerName) == False and TimeCount < TimeThreshold :
#    time.sleep(60*5)
#    TimeCount = TimeCount + 60*5
    

# In[1]:



import sys
sys.path.append('/home/sancere/anaconda3/envs/tensorflowGPU/lib/python3.6/site-packages/')

import csbdeep 

import os
#To run the prediction on the GPU, else comment out this line to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.models import Config, CARE
from csbdeep.utils import Path
import glob
from tifffile import imread
from Functions import save_tiff_imagej_compatible



# **Movie 1**

# In[2]:


basedir = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/a_maugarny-cales/20200907_FRAP/Stitched/a/Without_Contrast_Change/'
basedirMaskResults = '/run/user/1000/gvfs/smb-share:server=isiserver.curie.net,share=u934/equipe_bellaiche/a_maugarny-cales/20200907_FRAP/Stitched/a/Without_Contrast_Change/Masks/'


# In[3]:


Model_Dir = '/run/media/sancere/DATA1/Lucas_Model_to_use/Mask_Generator/'
ModelName = 'Masks_Generator_bin1_onlyequalized'

model = CARE(config = None, name = ModelName, basedir = Model_Dir)


# In[4]:


Path(basedirMaskResults).mkdir(exist_ok = True)


# In[5]:


Raw_path = os.path.join(basedir, '*tif') #be careful tif TIF 
axes = 'YX'
filesRaw = glob.glob(Raw_path)
filesRaw.sort

for fname in filesRaw:
        x = imread(fname)
        print('Saving file' +  basedirMaskResults + '%s_' + os.path.basename(fname))
        mask = model.predict(x, axes, n_tiles = (1, 2)) 
        Name = os.path.basename(os.path.splitext(fname)[0])
        #png.from_array(mask, mode="L").save(basedirMaskResults + Name +'.png') 
        save_tiff_imagej_compatible((basedirMaskResults + Name), mask, axes)
        
        
        

# In[6]:       
        
from csbdeep.utils import Path

TriggerName = '/home/sancere/NextonDisk_1/TimeTrigger/TTMaska'
Path(TriggerName).mkdir(exist_ok = True)