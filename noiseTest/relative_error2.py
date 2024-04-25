import numpy as np
import math
import pandas as pd
import random
import time
import os

os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions
from pathlib import Path
import heaan_sdk as heaan

# set key_dir_path
key_file_path = Path('./keys_FTa')

# set parameter
params = heaan.HEParameter.from_preset("FTa")

# init context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_file_path,
    load_keys="all",
    generate_keys=True,
)

num_slots = context.num_slots
log_num_slot = context.log_slots 


def print_ctxt(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
            
error1 = []
relative1 = []
error2 = []
relative2 = []
for _ in range(1):

    m = [1]*num_slots
    msg = heaan.Block(context,encrypted=False, data=m)
    c1 = msg.encrypt(inplace=False)
    c2 = c1*c1
    
    # basic
    msg1 = c1.decrypt(inplace=False)
    res1=[0 for _ in range(num_slots)]

    for i in range(num_slots):
        res1[i]=msg1[i].real
    
    err_list1 = []
    relative_err_list1 = [] 
    for i in range(len(m)):    
        err1 = abs(res1[i] - m[i])
        relative_err1 = abs(res1[i] - m[i])/m[i]
        err_list1.append(err1)
        relative_err_list1.append(relative_err1)
    error1.append(sum(err_list1)/len(err_list1))
    relative1.append(sum(relative_err_list1)/len(relative_err_list1))
    
    # power
    msg2 = c2.decrypt(inplace=False)
    res2=[0 for _ in range(num_slots)]

    for i in range(num_slots):
        res2[i]=msg2[i].real
        
    err_list2 = []
    relative_err_list2 = [] 
    for i in range(len(m)):    
        err2 = abs(res2[i] - m[i])
        relative_err2 = abs(res2[i] - m[i])/m[i]
        err_list2.append(err2)
        relative_err_list2.append(relative_err2)
    error2.append(sum(err_list2)/len(err_list2))
    relative2.append(sum(relative_err_list2)/len(relative_err_list2))

err1 = np.array(error1)
rel_err1=np.array(relative1)
print("error1 :",np.mean(err1),'+-',np.std(err1))    
print("rel_error1 :",np.mean(rel_err1),'+-',np.std(rel_err1))    

err2 = np.array(error2)
rel_err2=np.array(relative2)
print("error2 :",np.mean(err2),'+-',np.std(err2))    
print("rel_error2 :",np.mean(rel_err2),'+-',np.std(rel_err2))  
