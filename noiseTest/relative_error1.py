import numpy as np
import math
import pandas as pd
import random
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"  # set the number of CPU threads to use for parallel regions
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

def to_list(c,logN=log_num_slot):
    # ctxt to list
    # 실수만 들어있는 list 반환 
    m = c.decrypt(inplace=False)
    res=[0 for _ in range(num_slots)]

    for i in range(num_slots):
        res[i]=m[i].real

    return res


def cal_err(ctxt):
    # calculate the relative err
    res=np.array(to_list(ctxt))

    err=(np.max(res)-np.min(res))/np.mean(res)

    print("sum ",np.mean(res))
    print("relate err",err)


def generate_random_number():
    # Return random floats in the half-open interval [0.0, 1.0).
    random_number = np.random.random_sample()
    return random_number

def left_rotate_reduce(context,data,gs,interval):

    m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    res = m0.encrypt()
    
    empty_msg= heaan.Block(context,encrypted = False)
    rot = empty_msg.encrypt(inplace=False)
    
    binary_list = []
    while gs > 1:
        if gs%2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs//2
    binary_list.append(gs)

    # print("1")
    # print_ctxt1(data,context.num_slots)
    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data
            # print("0")
            # print_ctxt1(tmp,context.num_slots)
            while ind < i:
                
                rot = tmp.__lshift__(s)
                # print("1")
                # print_ctxt1(rot,context.num_slots)
                # check_boot()
                tmp = tmp + rot
                # print("2")
                # print_ctxt1(tmp,context.num_slots)
                s = s*2
                ind = ind+1
            if sdind > 0:
                tmp = tmp.__lshift__(sdind)
            # print("3")
            # print_ctxt1(tmp,context.num_slots)
            res = res + tmp
            # print("4")
            # print_ctxt1(res,context.num_slots)
            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    
    return res

def lshift(test_list, n):
  return test_list[n:] + test_list[:n]

def left_rotate_reduce_plain(data,gs,interval):

    # m0 = heaan.Block(context,encrypted = False, data = [0]*context.num_slots)
    # res = m0.encrypt()
    res = [0]*num_slots
    rot = [0]*num_slots
    # empty_msg= heaan.Block(context,encrypted = False)
    # rot = empty_msg.encrypt(inplace=False)
    
    binary_list = []
    while gs > 1:
        if gs%2 == 1:
            binary_list.append(1)
        else:
            binary_list.append(0)
        gs = gs//2
    binary_list.append(gs)

    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data # = list

            while ind < i:
                
                # rot = tmp.__lshift__(s)
                rot = lshift(tmp,s)
                tmp = tmp + rot
                s = s*2
                ind = ind+1
            if sdind > 0:
                # tmp = tmp.__lshift__(sdind)
                tmp = lshift(tmp,sdind)
            res = res + tmp
            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    
    return res
# # |평문 연산 결과 - 혜안아이티 연산결과|/(평문 연산 결과)

sumgroup_rel_err = []


for _ in range(1):
    val=generate_random_number()

    m = [val]*num_slots
    msg = heaan.Block(context,encrypted=False, data=m)
    c1 = msg.encrypt(inplace=False)
    
    # sumgroup
    sum_ctxt = left_rotate_reduce(context,c1,num_slots,1)
    sum_li =np.array(to_list(sum_ctxt))
    
    tmp = [val]*num_slots
    sum_plain = left_rotate_reduce_plain(tmp,num_slots,1)
    print('sumgroup')
    print(len(sum_plain),len(sum_li))
    sum_err = abs(sum_plain-sum_li)/sum_plain
    sum_res = sum(sum_err)/len(sum_err)
    sumgroup_rel_err.append(sum_res)
    
    
sumgroup_rel_err=np.array(sumgroup_rel_err)
print("sumgroup_rel_err :",np.mean(sumgroup_rel_err),'+-',np.std(sumgroup_rel_err)) 
