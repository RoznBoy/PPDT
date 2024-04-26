import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import heaan_sdk as heaan
import pickle
import gzip
import json
import math
import natsort
import re
import random

##=====setting=======##

# set the number of CPU threads to use for parallel regions
os.environ["OMP_NUM_THREADS"] = "16" 

# set key_dir_path
key_file_path = Path('./keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_file_path,
    load_keys="all",
    generate_keys=False,
)
num_slot = context.num_slots
log_num_slot = context.log_slots

def save_metadata_json(csv, depth, json_path):
    df = pd.read_csv(csv)   
    # meta_data함수로 ndata,n,d,t 파악하기
    ndata, d, n, t = meta_data(df)
    Metadata = {'ndata':ndata,
                'n':n,
                'd':d,
                't':t,
                'depth' : depth}
    
    # Metadata라는 이름으로 json 파일 저장(경로 확인!)
    with open(json_path + "Metadata.json", "w") as json_file:
        json.dump(Metadata, json_file, indent=2)
        
def meta_data(df):
    # 떨어진 column 값을 연속되게 바꿔 
    col = df.columns

    for cname in col:
        df[cname] = df[cname].astype('category')
        tmp_cat = np.sort(df[cname].unique())
        for j in range(len(tmp_cat)):
            tmp_cat[j] = j+1
        df[cname].values.set_categories = tmp_cat

    df = df.astype('int64')
    
    ndata = df.shape[0]
    d = len(col)-1
    n = find_max_cat_X(df)
    t = len(df['label'].unique())

    return ndata, d, n, t

def n_slot_func(gini):
    
    while gini%4 != 0:
        gini += 1    
      
    n_slot = 1.5*gini 
        
    return int(n_slot)

def left_rotate_reduce(data,interval,gs):
    # data = Block

    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
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

    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data
            while ind < i:
                rot = tmp.__lshift__(s)
                tmp = tmp + rot
                s = s*2
                ind = ind+1
            if sdind > 0:
                tmp = tmp.__lshift__(sdind)
            res = res + tmp
            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    
    return res

def right_rotate_reduce(data,interval,gs):
    # data = Block
   
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
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

    i = len(binary_list)-1
    sdind = 0
    while i >= 0:
        if binary_list[i] == 1:
            ind = 0
            s = interval
            tmp = data
            while ind < i:
                rot = tmp.__rshift__(s)
                tmp = tmp + rot
                s = s*2
                ind = ind+1
            if sdind > 0:
                tmp = tmp.__rshift__(sdind)
            res = res + tmp
            sdind = sdind + s
        i = i - 1            

    del  rot, tmp
    
    return res

def check_boot(x):
    if x.level==3:
        x.bootstrap()
    elif x.level<3:
        print('ciphertext died...')
        exit(0)
    return x

class Node(object):
    '''
    Representing a node in a decision tree
    Need to take the maximum number of possible elements in a variable
    and the number of variables in volved
    and the maximum depth of tree
    It manages the current depth
    '''
    ## class member variables
    ## ca,cv: list of ciphertexts 
    ca = [] ## indicating which variable is used for classification in this node 
    cv = [] ## indicating which value goes to left and right. 
    id = "" ## String type. Maintain the position in the tree
    cy = None ## Will be a ciphertext object having the return value
    by = None ## Will be a ciphertext object indicating the existence of cy
    ####
    left = None ## will be the left child
    right = None ## right child
    id = None ## position in the tree
    ###
    maxn = 0 ## maximum number of values that a variable can have
    d = 0 ## the number of participating variables
    
    def __init__(self, p_maxn: int, p_d: int):
        maxn = p_maxn
        d = p_d

def find_max_cat_X(df):
    col = df.columns
    max_values = []
    for i in col.drop('label'):
        max_values.append(max(df[i]))
    max_x = int(max(max_values))
    return max_x
    
## nsp: number of data set 
## n_comp: multiple of 4
## n_comp * 1.5 * n_set < num_slot 
## every set has 1/2*n_comp empty slots
## n_slot > n_comp * 1.5
def findMin4Many(c, d, n, n_comp, n_slot, n_set):
    if (n_comp==1): 
        # print("findMin4 ends..")
        # print_ctxt(c,1)
        return c
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    
    if (n_comp % 4 !=0):
        i=n_comp
        m=[0] * (num_slot) 
        while (i % 4 !=0):
            for j in range(n_set): 
                m[i+j*(n_slot)]=1.0000
            i+=1
        n_comp=i
        msg = heaan.Block(context,encrypted = False, data = m)
        c = msg + c
        
    # print("Min 1 ",n_comp)
    # print_ctxt(c, n_comp)

    ## Divide c by 4 chunk
    m = [0]*(num_slot)
    for j in range(n_set): 
        for i in range(n_comp//4):
            m[j*(n_slot)+i]=1
    msg1 = heaan.Block(context,encrypted = False, data = m)

    ca = c * msg1

    ctmp1 = c.__lshift__(n_comp//4)

    cb = ctmp1 * msg1
    # check_boot(cb)

    ctmp1 = c.__lshift__(n_comp//2)

    cc = ctmp1 * msg1
    # check_boot(cc)

    ctmp1 = c.__lshift__(n_comp*3//4)

    cd = ctmp1 * msg1
    # check_boot(cd)

    c1 = ca - cb
    c2 = cb - cc
    c3 = cc - cd
    c4 = cd - ca
    c5 = ca - cc
    c6 = cb - cd

    ctmp1 = c2.__rshift__(n_comp//4)
    ctmp1 = c1 + ctmp1

    ctmp2 = c3.__rshift__(n_comp//2)
    ctmp1 = ctmp2 + ctmp1

    ctmp2 = c4.__rshift__(n_comp*3//4)
    ctmp1 = ctmp2 + ctmp1

    ctmp2 = c5.__rshift__(n_comp)
    ctmp1 =  ctmp2 + ctmp1

    ctmp2 = c6.__rshift__(5*n_comp//4)
    ctmp1 = ctmp2 + ctmp1
    
    # print("Min 2 ")
    # print_ctxt(ctmp1, n_comp)

    start = time.time()
    c0 = ctmp1.sign(inplace = True, log_range=0)
    print(n_set, 'Min4Many approx sign: %.5f sec' %(time.time()-start))
    c0.bootstrap()
    # print("Min 3 ")
    # print_ctxt(c0, n_comp)

    c0_c = c0
    mkall = heaan.Block(context,encrypted = False, data = [1]*(num_slot))

    c0 = mkall + c0
    c0 = 0.5 * c0
    check_boot(c0)

    ## Making equality ciphertext 
    ceq = c0_c * c0_c
    check_boot(ceq)
    ceq = ceq.__neg__()
    ceq = mkall + ceq
    # print("step 6")
    ## Step 6..
    mk1=msg1
    mk2=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk3=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk4=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk5=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk6=heaan.Block(context,encrypted = False, data = [0]*num_slot)

    

    # print("step 6-1")
    m = [0]*(num_slot)
    for j in range(n_set):
        for i in range(n_comp//4,n_comp//2):
            m[j*(n_slot)+i]=1
    mk2 = heaan.Block(context,encrypted = False, data = m)
    
    m = [0]*(num_slot)
    for j in range(n_set):
        for i in range(n_comp//2,(3*n_comp)//4):
            m[j*(n_slot)+i]=1
    mk3 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(n_set):
        for i in range((3*n_comp)//4,n_comp):
            m[j*(n_slot)+i]=1
    mk4 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(n_set):
        for i in range(n_comp,(5*n_comp)//4):
            m[j*(n_slot)+i]=1
    mk5 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(n_set):
        for i in range((5*n_comp)//4,(3*n_comp)//2):
            m[j*(n_slot)+i]=1
    mk6 = heaan.Block(context,encrypted = False, data = m)

    ## Step 7
    # print("step 7")
    c_neg = c0
    c_neg = c0.__neg__()
 
    c_neg = mkall + c_neg ## c_neg = 1-c0 

    ### When min=a    
    c0n = c0
    ctmp1 = c_neg * mk1
    check_boot(ctmp1)
    ctxt=c0n

    c0=ctxt
    ctmp2 = c0 * mk4
    check_boot(ctmp2)

    ctmp2 = ctmp2.__lshift__((3*n_comp)//4)
    cda = ctmp2 ## (d>a)
    
    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)

    ctmp2 = c_neg * mk5

    ctmp2 = ctmp2.__lshift__(n_comp)
    ## cca
    cca = ctmp1 * ctmp2
    check_boot(cca)
    ### Need to add many cases to cca 

    ## Min=b
    ctmp1 = c0 * mk1 
    ctmp2 = c_neg * mk2
    ctmp2 = ctmp2.__lshift__(n_comp//4)
    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)

    ctmp2 = c_neg * mk6
    ctmp2 = ctmp2.__lshift__(n_comp*5//4)
    ccb = ctmp1 * ctmp2
    check_boot(ccb)

    ## Min=c
    ctmp1 = c0 * mk2
    ctmp1 = ctmp1.__lshift__(n_comp//4)
    cbc = ctmp1 ## (b>c)

    ctmp2 = c_neg * mk3
    ctmp2 = ctmp2.__lshift__(n_comp//2)
    ctmp1 = ctmp2 * ctmp1
    ctmp2 = c0 * mk5
    ctmp2 = ctmp2.__lshift__(n_comp)
    ccc = ctmp1 * ctmp2
    check_boot(ccc)

    ## Min=d
    ctmp1 = c0 * mk3
    ctmp1 = ctmp1.__lshift__(n_comp//2)
    ctmp2 = c_neg * mk4
    ctmp2 = ctmp2.__lshift__(3*n_comp//4)
    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)

    ctmp2 = c0 * mk6
    ctmp2 = ctmp2.__lshift__(5*n_comp//4)

    ccd = ctmp1 * ctmp2
    check_boot(ccd)
   
    ## debugging (Non case) 
    ## Special case 1 
    ctmp1 = cca.__neg__()
    ctmp1 = mk1 + ctmp1
    ctmp2 = ccb.__neg__()
    ctmp2 = mk1 + ctmp2
    ccs1 = ctmp1 * ctmp2
    check_boot(ccs1)

    ctmp1 = ccc.__neg__()
    ctmp1 = mk1 + ctmp1
    ctmp2 = ccd.__neg__()
    ctmp2 = mk1 + ctmp2

    ccs1 = ctmp1 * ccs1
    ccs1 = ctmp2 * ccs1
    check_boot(ccs1)
    ## if ccs1 = 1, a random number is selected
    ccs_sel = random.randrange(1,5)
    if (ccs_sel == 1):
        ccs1 = ccs1 * ca
    elif (ccs_sel==2):
        ccs1 = cb * ccs1
    elif (ccs_sel==3):
        ccs1 = cc * ccs1
    else:
        ccs1 = cd * ccs1
    check_boot(ccs1)
    
    # print("Min 4 ")
    # print_ctxt(ccs1, n_comp)

    # print("step 8")

    cca = ca * cca
    ccb = cb * ccb
    ccc = cc * ccc
    ccd = cd * ccd

    cout = cca
    cout = ccb + cout
    cout = ccc + cout
    cout = ccd + cout
    cout = ccs1 + cout   
    check_boot(cout) 

    ### Going to check equality
    # print("step 9")
    ceq_ab = ceq * mk1
    check_boot(ceq_ab)

    ceq_bc = ceq.__lshift__((n_comp)//4)
    ceq_bc = mk1 * ceq_bc
    check_boot(ceq_bc)

    ceq_cd = ceq.__lshift__((n_comp)//2) 
    ceq_cd = mk1 * ceq_cd
    check_boot(ceq_cd)

    ceq_da = ceq.__lshift__((n_comp)*3//4)
    ceq_da = mk1 * ceq_da
    check_boot(ceq_da)

    ## Checking remaining depth
    ncda = cda
    ncda = ncda.__neg__()
    ncda = mk1 + ncda

    ncbc = cbc
    ncbc = ncbc.__neg__()
    ncbc = mk1 + ncbc

    ctmp2 = ceq_ab * ceq_bc
    ctmp1 = ctmp2 * cda
    c_cond3 = ctmp1
    
    print("Min 5 ")
    print_ctxt(c_cond3, n_comp)

    ## (b=c)(c=d)(1-(d>a))
    ctmp1 = ceq_bc * ceq_cd
    ctmp1 = ncda * ctmp1
    c_cond3 = ctmp1 + c_cond3

    ## (c=d)(d=a)(b>c)
    ctmp1 = ceq_cd * ceq_da
    ctmp1 = cbc * ctmp1
    c_cond3 = ctmp1 + c_cond3

    ## (d=a)(a=b)(1-(b>c))
    ctmp1 = ceq_ab * ceq_da
    ctmp1 = ncbc * ctmp1
    c_cond3 = ctmp1 + c_cond3
  
    c_cond4 = ctmp2 * ceq_cd
    print("Min 6 ")
    print_ctxt(c_cond4, n_comp)

    # print("step 10")

    c_tba = c_cond3 * 0.333333333
    c_tba = mkall + c_tba
    c_cond4 = 0.333333333 * c_cond4
    c_tba = c_cond4 + c_tba
    cout = c_tba * cout
    check_boot(cout)
    
    ##print("current cmin output----", n_comp, n_slot, n_set)
    ##print_ctxt(cout,dec,sk,logN-1,n_slot)

    return findMin4Many(cout, d, n, n_comp//4, n_slot, n_set)

def findMinPosMany(c,d,n,n_comp,n_slot,n_set):
    start = time.time()
    cmin = findMin4Many(c,d,n,n_comp,n_slot,n_set)
    print(n_set, 'findMin4Many time: %.5f sec' %(time.time()-start))
    
    print("in findminposmany cmin: ")
    print_ctxt(cmin, n_comp)
    
    m = [0] * (num_slot)
    m1 = [0] * (num_slot)
    for i in range(n_set):
        m[i*(n_slot)]=1
        for j in range(math.ceil(n_comp/4)*4,n_slot):
            m1[i*(n_slot)+j]=1

    m_ = heaan.Block(context,encrypted = False, data = m)
    cmin = m_ * cmin

    m1_ = heaan.Block(context,encrypted = False, data = m1)
    c = m1_ + c
    

    ## Extension
    ## Suppose slot size of 2^N form
    ## Copy n Set -> cmin 의 최소값을 n_comp 개만큼만 복사 
    ## n_comp binary encoding
    bs = math.ceil(n_slot+0.1)
    be = n_slot
    pt = 0
    na = [0]*bs
    while(be>1):
       na[pt] = be % 2
       pt+=1
       be = be//2
    na[pt]=be
    pt+=1
    cmin_a = []
    ctmp = cmin
    cmin_a.append(ctmp)
    for i in range(pt-1):
        ctmp = cmin_a[i].__rshift__(2**i)
        ctmp = cmin_a[i] + ctmp
        cmin_a.append(ctmp)
    ## Building CopynSet 
    ct=0
    c_ret = None
    for i in range(pt):
        if (na[i]==1):
            if (ct==0):
                c_ret = cmin_a[i]
                ct+=2**i
            else:
                ctmp = cmin_a[i]
                ctmp = ctmp.__rshift__(ct)
                c_ret = ctmp + c_ret
                ct+=2**i

    ## c_ret has all c_mins
    c = c - c_ret

    c = c - 1/2**16
    check_boot(c)
    ### Here...
    # ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    # ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    # c.bootstrap()
    start = time.time()
    c_red = c.sign(inplace = True, log_range=0)
    print(n_set, 'MinPosMany approx sign: %.5f sec' %(time.time()-start))
    ##print("after approx sign: ")
    ##print_ctxt(c_red, dec, sk, logN, n*d)
    c_red.bootstrap()
    c_red = c_red.__neg__()
    c_red = 1 + c_red
    c_red = 0.5 * c_red
    check_boot(c_red)

    ### Need to generate a rotate sequence to choose 1 in a random position   
    c_out = selectRandomOnePosMany(c_red,d,n,n_comp,n_slot,n_set)

    del ctmp, cmin, c_red, c_ret 
    return c_out

def selectRandomOnePosMany(c_red,d,n,ndata,nslot,nset):
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    c_sel = m0.encrypt()
    
    rando = np.random.permutation(ndata)
    ctmp1 = c_red

    m0_ = [0]*(num_slot)
    for i in range(nset):
        m0_[i*(nslot)]=1
    m0 = heaan.Block(context,encrypted = False, data = m0_)
    
    for l in rando:
        if (l>0):
            ctmp1 = ctmp1.__lshift__(l)
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            check_boot(ctmp1)

            ctmp2 = ctmp1 * m0
            ctmp1 = ctmp1.__rshift__(l)
            c_sel = c_sel + ctmp2
            check_boot(c_sel)
        else:
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * m0
            c_sel = c_sel + ctmp2
            check_boot(c_sel)

    return ctmp1

def findMax4Many(c, d, n, ndata, nslot, nset):
    if (ndata==1): return c
    ##print("cinx level:",c.level)

    ## divide ctxt by four chunks
    if (ndata % 4 !=0):
        i=ndata
        m=[0] * (num_slot)
        while (i % 4 !=0):
            for j in range(nset):
                m[i+j*nslot]=-0.02
            i+=1
        ndata=i
        msg = heaan.Block(context,encrypted = False, data = m)
        c = msg + c

    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
 
    ## Divide c by 4 chunk
    m = [0]*(num_slot)
    for j in range(nset):
        for i in range(ndata//4):
            m[j*nslot+i]=1
    msg1 = heaan.Block(context,encrypted = False, data = m)

    ca = c * msg1

    ctmp1 = c.__lshift__(ndata//4)

    cb = ctmp1 * msg1
    # check_boot(cb)

    ctmp1 = c.__lshift__(ndata//2)

    cc = ctmp1 * msg1
    # check_boot(cc)

    ctmp1 = c.__lshift__(ndata*3//4)

    cd = ctmp1 * msg1

    c1 = ca - cb
    c2 = cb - cc
    c3 = cc - cd
    c4 = cd - ca
    c5 = ca - cc
    c6 = cb - cd

    ctmp1 = c2.__rshift__(ndata//4)
    ctmp1 = c1 + ctmp1

    ctmp2 = c3.__rshift__(ndata//2)
    ctmp1 = ctmp2 + ctmp1

    ctmp2 = c4.__rshift__(ndata*3//4)
    ctmp1 = ctmp2 + ctmp1

    ctmp2 = c5.__rshift__(ndata)
    ctmp1 =  ctmp2 + ctmp1

    ctmp2 = c6.__rshift__(5*ndata//4)
    ctmp1 = ctmp2 + ctmp1
    
    start = time.time()
    c0 = ctmp1.sign(inplace = True, log_range=0)
    print(nset, 'Min4Many approx sign: %.5f sec' %(time.time()-start))
    c0.bootstrap()
    
    c0_c = c0
    mkall = heaan.Block(context,encrypted = False, data = [1]*(num_slot))

    c0 = mkall + c0
    c0 = 0.5 * c0
    check_boot(c0)

    ## Making equality ciphertext 
    ceq = c0_c * c0_c
    check_boot(ceq)
    ceq = ceq.__neg__()
    ceq = mkall + ceq
    # print("step 6")
    ## Step 6..
    mk1=msg1
    mk2=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk3=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk4=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk5=heaan.Block(context,encrypted = False, data = [0]*num_slot)
    mk6=heaan.Block(context,encrypted = False, data = [0]*num_slot)



    # print("step 6-1")
    m = [0]*(num_slot)
    for j in range(nset): 
        for i in range(ndata//4,ndata//2):
            m[j*(nslot)+i]=1
    mk2 = heaan.Block(context,encrypted = False, data = m)
    
    m = [0]*(num_slot)
    for j in range(nset): 
        for i in range(ndata//2,(3*ndata)//4):
            m[j*(nslot)+i]=1
    mk3 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(nset): 
        for i in range((3*ndata)//4,ndata):
            m[j*(nslot)+i]=1
    mk4 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(nset): 
        for i in range(ndata,(5*ndata)//4):
            m[j*(nslot)+i]=1
    mk5 = heaan.Block(context,encrypted = False, data = m)

    m = [0]*(num_slot)
    for j in range(nset): 
        for i in range((5*ndata)//4,(3*ndata)//2):
            m[j*(nslot)+i]=1
    mk6 = heaan.Block(context,encrypted = False, data = m)

    ## Step 7
    # print("step 7")
    c_neg = c0
    c_neg = c0.__neg__()
 
    c_neg = mkall + c_neg ## c_neg = 1-c0 

    ### When max=a
    ## ctmp1 = a>b
    c0n = c0
    ctmp1 = c0n * mk1
    ctxt=c0n

    c_ab = ctmp1
  
    ## ctmp2 = a>d
    c0=ctxt
    ctmp2 = c_neg * mk4
    check_boot(ctmp2)
    
    ctmp2 = ctmp2.__lshift__((3*ndata)//4)

    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)
    ## ctmp2 = a>c

    ctmp2 = c0 * mk5
    check_boot(ctmp2)
    ctmp2 = ctmp2.__lshift__(ndata)

    ## cca
    cca = ctmp1 * ctmp2
    check_boot(cca)

    ## Min=b
    ctmp1 = c_neg * mk1
    ctmp2 = c0 * mk2
    ctmp2 = ctmp2.__lshift__(ndata//4)
    c_bc = ctmp2
    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)

    ctmp2 = c0 * mk6
    ctmp2 = ctmp2.__lshift__(ndata*5//4)
    ccb = ctmp1 * ctmp2
    check_boot(ccb)

    ## Min=c
    ctmp1 = c_neg * mk2
    ctmp1 = ctmp1.__lshift__(ndata//4)
    ctmp2 = c0 * mk3
    ctmp2 = ctmp2.__lshift__(ndata//2)
    c_cd = ctmp2
    ctmp1 = ctmp2 * ctmp1
    ctmp2 = c_neg * mk5
    ctmp2 = ctmp2.__lshift__(ndata)
    ccc = ctmp1 * ctmp2
    check_boot(ccc)

    ## Min=d
    ctmp1 = c_neg * mk3
    ctmp1 = ctmp1.__lshift__(ndata//2)
    ctmp2 = c0 * mk4
    ctmp2 = ctmp2.__lshift__(3*ndata//4)
    cda = ctmp2
    ctmp1 = ctmp2 * ctmp1
    check_boot(ctmp1)

    ctmp2 = c_neg * mk6
    ctmp2 = ctmp2.__lshift__(5*ndata//4)
    ccd = ctmp1 * ctmp2
    check_boot(ccd)

    cca = ca * cca
    ccb = cb * ccb
    ccc = cc * ccc
    ccd = cd * ccd
    
    cout = cca
    cout = ccb + cout
    cout = ccc + cout
    cout = ccd + cout
    check_boot(cout) 

    ### Going to check equality
    #print("step 9")
    cneq = ceq.__neg__()
    cneq = mkall + cneq ## cneq = 1-ceq
    cneq_da = ceq.__lshift__((ndata)*3//4)
    cneq_da = mk1 * cneq_da
    check_boot(cneq_da)

    cneq_bc = cneq.__lshift__((ndata)//4)
    cneq_bc = mk1 * cneq_bc
    check_boot(cneq_bc)

    ceq_ab = ceq * mk1
    check_boot(ceq_ab)

    ceq_bc = ceq.__lshift__((ndata)//4)
    ceq_bc = mk1 * ceq_bc
    check_boot(ceq_bc)

    ceq_cd = ceq.__lshift__((ndata)//2) 
    ceq_cd = mk1 * ceq_cd
    check_boot(ceq_cd)

    ceq_da = cneq_da.__neg__()
    ceq_da = mk1 + ceq_da


    ## Need to check (a=b)(b=c)(c>d)
    ctmp2 = ceq_ab * ceq_bc
    ctmp1 = ctmp2 * c_cd
    c_cond3 = ctmp1

    ## (b=c)(c=d)(d>a)
    ctmp1 = ceq_bc * ceq_cd
    ctmp1 = cda * ctmp1
    c_cond3 = ctmp1 + c_cond3

    ## (c=d)(d=a)(a>b)
    ctmp1 = ceq_cd * ceq_da
    ctmp1 = c_ab * ctmp1
    c_cond3 = ctmp1 + c_cond3

    ## (a=b)(d=a)(b>c)
    ctmp1 = ceq_ab * ceq_da
    ctmp1 = c_bc * ctmp1
    c_cond3 = ctmp1 + c_cond3

    c_cond4 = ctmp2 * ceq_cd

    c_tba = c_cond3 * 0.333333333
    c_tba = mkall + c_tba
    ctmp1 = c_cond4 + mkall  
    c_tba = ctmp1 * c_tba
    cout = c_tba * cout
    check_boot(cout)

    #print("current max y output----", ndata, nslot, nset)
    #print_ctxt(cout,dec,sk,logN-1,nslot)

    return findMax4Many(cout, d, n, ndata//4,nslot,nset)

def findMaxPosMany(c,d,n,ndata,nslot,nset):
    start = time.time()
    cmax = findMax4Many(c,d,n,ndata,nslot,nset)
    print(nset, 'findMax4Many time: %.5f sec' %(time.time()-start))
    
    print("in findMaxposmany cmin: ")
    print_ctxt(cmax, ndata)
    
    ## Extension
    ## Suppose slot size of 2^N form
    m = [0] * (num_slot)
    m1 = [0] * (num_slot)
    for i in range(nset):
        m[i*nslot]=1
        for j in range(math.ceil(ndata/4)*4,nslot):
            m1[i*(nslot)+j]=-0.02


    m_ = heaan.Block(context,encrypted = False, data = m)
    cmax = m_ * cmax

    m1_ = heaan.Block(context,encrypted = False, data = m1)
    c = m1_ + c

    ## Copy n Set -> cmax 의 값을 n_comp 개만큼만 복사
    ## n_comp binary encoding
    bs = math.ceil(nslot+0.1)
    be = nslot
    pt = 0
    na = [0]*bs
    while(be>1):
       na[pt] = be % 2
       pt+=1
       be = be//2
    na[pt]=be
    pt+=1
    cmax_a = []
    ctmp = cmax
    cmax_a.append(ctmp)
    for i in range(pt-1):
        ctmp = cmax_a[i].__rshift__(2**i)
        ctmp = cmax_a[i] + ctmp
        cmax_a.append(ctmp)
    ## Building CopynSet
    ct=0
    c_ret = None
    for i in range(pt):
        if (na[i]==1):
            if (ct==0):
                c_ret = cmax_a[i]
                ct+=2**i
            else:
                ctmp = cmax_a[i]
                ctmp = ctmp.__rshift__(ct)
                c_ret = ctmp + c_ret
                ct+=2**i




    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    c = c - c_ret

    c = c + 1/2**16
    check_boot(c)
    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    # c.bootstrap()
    start = time.time()
    c_red = c.sign(inplace = True, log_range=0)
    print(nset, 'MinPosMany approx sign: %.5f sec' %(time.time()-start))
    c_red.bootstrap()
    
    c_red = 1 + c_red
    c_red = 0.5 * c_red
    check_boot(c_red)

    ### Need to generate a rotate sequence to choose 1 in a random position
    # c_out = heaan.Ciphertext(context)
    # heaan.math.approx.discrete_equal_zero(he, c_red, c_out)
    c_out = selectRandomOnePosMany(c_red,d,n,ndata,nslot,nset)
    # print("selRandOne cmin:",c_out.level)
    # print("findMaxPos result :")
    # print_ctxt(c_out,dec,sk,logN,ndata)

    del ctmp, cmax, c_red, c_ret 

    return c_out

def load_ctxt(fn_list,ctxt_path): ## returns the list of ciphertexts loaded
    out_cdict={}
    for cname in fn_list:
        m = heaan.Block(context,encrypted = False)
        ctxt = m.encrypt()
        ctxt.load(ctxt_path+cname+".ctxt")
        out_cdict[cname]=ctxt
    return out_cdict

def make_file_list(ctxt_path):
    f_list = os.listdir(ctxt_path)
    file_list = []
    for i in f_list:
        if 'total' not in i:
            file_list.append(re.sub('.ctxt','',i))
    file_list = natsort.natsorted(file_list)
    return file_list

def findMaxY(cdict,d,n,t,ndata):
    pow_2_ndata = _get_smallest_pow_2(ndata)
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1/(ndata)]+[0]*(num_slot-1))
    value = m0.encrypt()

    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = cdict[label]
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = rot + tmp
            r*=2
        tmp = m1 * tmp
        rot = tmp.__rshift__(i-1)
        value = rot + value
    #print("value:")
    #print_ctxt(value,dec,sk,logN-1,t)

    ## The returning poisition will be in [0,t-1]
    return value

def findMaxY_big(L,ctxt_path,t,ndata):
    #print("findMaxY...")
    pow_2_ndata = _get_smallest_pow_2(ndata)
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1/(ndata)]+[0]*(num_slot-1))
    empty_msg= heaan.Block(context,encrypted = False)
    value = m0.encrypt()
    tmp = empty_msg.encrypt()

    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = tmp.load(ctxt_path+L.id+'/'+label+'.ctxt')
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = rot + tmp
            r*=2
        tmp = m1 * tmp
        rot = tmp.__rshift__(i-1)
        value = rot + value
    # print("======================= value =======================")
    # print_ctxt(value,dec,sk,logN-1,t)

    return value

def print_ctxt(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        if m[i].real > 0.9:
            print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
            
def _get_smallest_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def make_ca_cv(cmin,d,n):
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)

    cmin = left_rotate_reduce(cmin,1,n)

    ## making c_a, c_v
    m01 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    
    #print("============================= making c_a c_v ====================")
    c_a = m0.encrypt()
    c_v = m0.encrypt()
    for i in range(d):
        if (i>0):
            ctmp = cmin.__lshift__(n*i)
        else:
            ctmp = cmin

        tmp = ctmp * m01

        bin_ext = right_rotate_reduce(tmp,1,n)
        bin_ext  = ctmp * bin_ext
        c_v = bin_ext * c_v  
        
        ctmp = m01 * ctmp
        ctmp = ctmp.__rshift__(i)
        c_a = ctmp + c_a
    #print("ca")
    #print_ctxt(c_a,dec,sk,logN,d) 

    #print("cv")
    #print_ctxt(c_v,dec,sk,logN,n)
    #print('====================== ca cv end =========================')
    return c_a,c_v

def DT_Learn(L,cdict,cmin,cy,d,n,t,ndata,model_path):
    pow_2_ndata = _get_smallest_pow_2(ndata)
    
    start = time.time()
    L.ca, L.cv = make_ca_cv(cmin,d,n)
    print(L.id, 'Find ca cv time: %.5f sec' %(time.time()-start))
    L.ca = L.ca.save(model_path+L.id+"_ca.ctxt")
    L.cv = L.cv.save(model_path+L.id+"_cv.ctxt")
        
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    mt = heaan.Block(context,encrypted = False, data = [1/(t*pow_2_ndata)]+[0]*(num_slot-1))

    Lcy = m0.encrypt()
    
    start = time.time()
    for i in range(t):
        tmp = cy.__lshift__(i)
        tmp = tmp * m1
        tmp = (i+1) * tmp
        Lcy = tmp + Lcy
    L.cy = Lcy
    print(L.id, 'cy time: %.5f sec' %(time.time()-start))

    # cacc = m0.encrypt()

    # start = time.time()
    # cacc = left_rotate_reduce(cy,1,t)
    # cacc = m1 * cacc

    # L.by = cacc

    value = m0.encrypt()
    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = cdict[label]
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = tmp + rot
            r*=2
        value = tmp + value
    tmp = value * mt

    inverse_start = time.time()
    sample_inverse = tmp.inverse(greater_than_one = False)
    print(L.id, 'sample inverse time: %.5f sec' %(time.time()-inverse_start))
    sample_inverse = tmp * sample_inverse
    L.by = sample_inverse
    print(L.id, 'by time: %.5f sec' %(time.time()-start))

    print('ca: ')
    print_ctxt(L.ca,d)
    print('cv: ')
    print_ctxt(L.cv,n)
    print('cy: ')
    print_ctxt(L.cy,t)
    print('by: ')
    print_ctxt(L.by,1)
    
    return L.ca,L.cv,L.cy,L.by

def DT_Learn_big(L,cmin,cy,d,n,t,ndata,model_path,ctxt_path):
    pow_2_ndata = _get_smallest_pow_2(ndata)

    start = time.time()
    L.ca, L.cv = make_ca_cv(cmin,d,n,ndata)
    print(L.id, 'Find_ca_cv time: %.5f sec' %(time.time()-start), flush = True)
    L.ca.save(model_path+L.id+"_ca.ctxt")
    L.cv.save(model_path+L.id+"_cv.ctxt")
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    mt = heaan.Block(context,encrypted = False, data = [1/(t*pow_2_ndata)]+[0]*(num_slot-1))
    empty_msg = heaan.Block(context,encrypted = False)
    
    Lcy = m0.encrypt()
    start = time.time() # evaluation으로 넘겨주기 위한 cy
    for i in range(t):
        tmp = cy.__lshift__(i)
        tmp = tmp * m1
        tmp = (i+1) * tmp
        Lcy = tmp + Lcy
    L.cy = Lcy
    print(L.id, 'cy time: %.5f sec' %(time.time()-start))

    # L.cy.to_device()
    # # print('DT_Learn 2', flush = True)
    # cacc = heaan.Ciphertext(context)
    # enc.encrypt(m0,kpack,cacc)
    # cacc.to_device()

    # start = time.time()
    # eval.left_rotate_reduce(cy,1,t,cacc)
    # mult(cacc,m1,cacc,eval)

    value = m0.encrypt()
    tmp = empty_msg.encrypt()
    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = tmp.load(ctxt_path+L.id+'/'+label+'.ctxt')
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = tmp + rot
            r*=2
        value = tmp + value
    tmp = value * mt
    # print('value: ', flush = True)
    # print_ctxt(value,dec,sk,logN-1,10)
    # mult(value, value, value, eval)
    # print('value: 2', flush = True)
    # print_ctxt(value,dec,sk,logN-1,10)
    # print('DT_Learn 4', flush = True)
    #print('Sample: ')
    #print_ctxt(sample,dec,sk,logN-1,1)
    inverse_start = time.time()
    sample_inverse = tmp.inverse(greater_than_one = False)
    print(L.id, 'sample inverse time: %.5f sec' %(time.time()-inverse_start))
    sample_inverse = tmp * sample_inverse
    sample_inverse = sample_inverse * sample_inverse
    L.by = sample_inverse
    print(L.id, 'by time: %.5f sec' %(time.time()-start))

    print('ca: ', flush = True)
    print_ctxt(L.ca,d)
    print('cv: ', flush = True)
    print_ctxt(L.cv,n)
    print('cy: ', flush = True)
    print_ctxt(L.cy,1)
    print('by: ', flush = True)
    print_ctxt(L.by,1)  
    
    return L.ca,L.cv,L.cy,L.by

def update_data(L,parent_id,cdict,d,n,t,ndata,model_path,ctxt_path):
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    empty_msg= heaan.Block(context,encrypted = False)
    
    ca = empty_msg.encrypt() 
    cv = empty_msg.encrypt() 
    ca = ca.load(model_path+parent_id+'_ca.ctxt')
    cv = cv.load(model_path+parent_id+'_cv.ctxt')

    Xd = m0.encrypt()
    child = m0.encrypt()

    mn = heaan.Block(context,encrypted = False, data = [1]*n + [0]*(num_slot-n))
    _cv = mn.encrypt()
    _cv = _cv - cv
    
    pow_2_ndata = _get_smallest_pow_2(ndata)
    idx = 'X'

    for i in range(d):
        rotate_ca = ca.__lshift__(i)
        rotate_ca = rotate_ca * m1

        rotate_ca = right_rotate_reduce(rotate_ca,1,pow_2_ndata*n)

        ctmp = cdict[idx].__lshift__(pow_2_ndata*n*i)
        ctmp = ctmp * rotate_ca
        Xd = Xd + ctmp

    if L.id[-1] == 'l':
        ## left branch   
        for j in range(n):
            rotate_cv = cv.__lshift__(j)
            rotate_cv = rotate_cv * m1

            rotate_cv = right_rotate_reduce(rotate_cv,1,pow_2_ndata)

            ctmp = Xd.__lshift__(pow_2_ndata*j)
            ctmp = ctmp * rotate_cv
            child = child + ctmp
            
    else:         
        ## right branch
        for j in range(n):
            rotate_cv = _cv.__lshift__(j)
            rotate_cv = rotate_cv * m1
            
            rotate_cv = right_rotate_reduce(rotate_cv,1,pow_2_ndata)

            ctmp = Xd.__lshift__(pow_2_ndata*j)
            ctmp = ctmp * rotate_cv
            child = child + ctmp

    r=1
    while (pow_2_ndata*r<num_slot):
        rot = child.__rshift__(pow_2_ndata*r)
        child = child + rot
        r*=2

    ctmp = cdict[idx] * child
    ctmp.save(ctxt_path+L.id+'/'+idx+'.ctxt')
    
    for i in range(1,t+1):
        ind='label_'+str(i)
        ctmp = cdict[ind] * child
        ctmp.save(ctxt_path+L.id+'/'+ind+'.ctxt')
            
    return 

def update_data_big(L,d,n,t,ndata,model_path,ctxt_path):
    parent_id = L.id[:len(L.id)-1]
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    empty_msg= heaan.Block(context,encrypted = False)
    
    ca = empty_msg.encrypt() 
    cv = empty_msg.encrypt() 
    ca = ca.load(model_path+parent_id+'_ca.ctxt')
    cv = cv.load(model_path+parent_id+'_cv.ctxt')

    Xd = m0.encrypt()
    child = m0.encrypt()

    mn = heaan.Block(context,encrypted = False, data = [1]*n + [0]*(num_slot-n))
    _cv = mn.encrypt()
    _cv = _cv - cv
    
    pow_2_ndata = _get_smallest_pow_2(ndata)
    n1 = int(num_slot//pow_2_ndata)
    d1 = int(np.ceil(d/n1))
    for j in range(n):
        Xd = m0.encrypt()
        for i in range(d):
            ctxt_num = int(i%d1)
            idx = 'X'+str(ctxt_num+1)+'_'+str(j+1)
            ctmp = empty_msg.encrypt()
            ctmp = ctmp.load(ctxt_path+parent_id+'/'+idx+'.ctxt')
            ctmp = ctmp.__lshift__(int(i//d1)*pow_2_ndata)

            rotate_ca = ca.__lshift__(i)
            rotate_ca = rotate_ca * m1
            rotate_ca = right_rotate_reduce(rotate_ca,1,pow_2_ndata)

            ctmp = ctmp * rotate_ca
            Xd = Xd + ctmp
 
        if L.id[-1] == 'l':
            rotate_cv = cv.__lshift__(j)
            rotate_cv = rotate_cv * m1
            rotate_cv = right_rotate_reduce(rotate_cv,1,pow_2_ndata)
            Xd = rotate_cv * Xd
            child = child + Xd
            
        else:
            rotate_cv = _cv.__lshift__(j)
            rotate_cv = rotate_cv * m1
            rotate_cv = right_rotate_reduce(rotate_cv,1,pow_2_ndata)
            Xd = rotate_cv * Xd
            child = child + Xd
        
    r=1
    while r < n1:
        rot = child.__rshift__(pow_2_ndata*r)
        child = child + rot
        r*=2

    for i in range(1,d1+1):
        for j in range(1,n+1):
            ind = 'X'+str(i)+'_'+str(j)
            ctmp = ctmp.load(ctxt_path+parent_id+'/'+ind+'.ctxt')
            ctmp = child * ctmp
            ctmp.save(ctxt_path+L.id+'/'+ind+'.ctxt')

    for i in range(1,t+1):
        ind='label_'+str(i)
        ctmp = ctmp.load(ctxt_path+parent_id+'/'+ind+'.ctxt')
        ctmp = child * ctmp
        ctmp.save(ctxt_path+L.id+'/'+ind+'.ctxt')
        
    return

def find2Gini(cdict,d,n,t,ndata):
    pow_2_ndata = _get_smallest_pow_2(ndata)
    scal = 1/ndata
    # print('scal', scal)
    X = 'X'
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    lctxt_sum = m0.encrypt()
    rctxt_sum = m0.encrypt()
    sqrt = m0.encrypt()
    gini = m0.encrypt()
    
    m = [1]*(pow_2_ndata*n)
    m_pow = heaan.Block(context,encrypted = False, data = m)

    xtmp = cdict[X]
    xtmp = scal * xtmp
    for i in range(1,t+1):
        label = 'label_'+str(i)

        tmp = cdict[label]
        ctxt_y = tmp * xtmp

        start = time.time()
        r = 1
        while (r<pow_2_ndata):
            rot = ctxt_y.__lshift__(r)
            ctxt_y = ctxt_y + rot
            r*=2
        print('rotate ctmp time: %.5f sec' %(time.time()-start))
 
        total_one = m0.encrypt()
        left = m0.encrypt()
        right = m0.encrypt()

        # left를 구하기 위함 -> X1이 X2 침범해사 마스킹 후 처리
        for k in range(d):
            ctmp = ctxt_y.__lshift__(pow_2_ndata*n*k)
            ctmp = m_pow * ctmp
            ###############################################################
            ########################## make total #########################
            ###############################################################
                 
            total_rot = left_rotate_reduce(ctmp,pow_2_ndata,n) 
            total_rot = m1 * total_rot 
            rot = total_rot.__rshift__(pow_2_ndata*n*k)
            total_one = rot + total_one
            
            ###############################################################
            ########################## make left ##########################
            ###########################################################
            
            left_rot = right_rotate_reduce(ctmp,pow_2_ndata,n)
            left_rot = m_pow * left_rot    
            ctmp = left_rot.__rshift__(pow_2_ndata*n*k)
            left = ctmp + left
            
        ###############################################################
        ###################### total을 전체에 복사 #####################
        ###############################################################
        total = right_rotate_reduce(total_one,pow_2_ndata,n)

        right = total - left
 
        lctxt_sum = left + lctxt_sum
        rctxt_sum = right + rctxt_sum

        left = left * left
        right = right * right
        sqrt = left + sqrt
        sqrt = right + sqrt
                   
    lctxt_sum = lctxt_sum * lctxt_sum
    rctxt_sum = rctxt_sum * rctxt_sum
    lctxt_sum = lctxt_sum + rctxt_sum
    lctxt_sum = lctxt_sum - sqrt
    
    if n*d < pow_2_ndata:
        m = [1]*(n*d) + [0]*(num_slot-n*d)
        m_gini = heaan.Block(context,encrypted = False, data = m)
        
        start = time.time()
        for i in range(n*d):
            rot = lctxt_sum.__lshift__((pow_2_ndata-1)*i)
            gini = rot + gini
        print('값 모으기 time: %.5f sec' %(time.time()-start))
            
        gini = m_gini * gini
    else:
        for i in range(n*d):
            m[i*pow_2_ndata] = 1
            m_gini = heaan.Block(context,encrypted = False, data = m)
            
            start = time.time()
            rot = lctxt_sum * m_gini
            rot = rot.__lshift__((pow_2_ndata-1)*i)
            gini = rot + gini
            print('값 모으기 time: %.5f sec' %(time.time()-start))
    
    # print('gini.level: ', gini.level)
    # print('지니 계수: ')
    #print_ctxt(gini,dec,sk,logN,d*n*pow_2_ndata)
    
    return gini

def calGini(L,ctxt_path,d,n,t,ndata):
    pow_2_ndata = _get_smallest_pow_2(ndata)
    scal = 1/ndata
    # scal = 1/np.sqrt(2*t*pow_2_ndata)
    d1 = num_slot//pow_2_ndata
    n1 = int(np.ceil(d/d1))
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    empty_msg= heaan.Block(context,encrypted = False)
    label_ctxt = empty_msg.encrypt()
    X_ctxt = empty_msg.encrypt()
    
    m = [0]*(num_slot)
    for i in range(n*d):
        m[i] = 1
    mt = heaan.Block(context,encrypted = False, data = m)
    
    m = [0]*(num_slot)
    for i in range(0,pow_2_ndata*d1,pow_2_ndata):
        m[i] = 1
    m_d1 = heaan.Block(context,encrypted = False, data = m)
    
    m = [0]*(num_slot)
    for i in range(0,pow_2_ndata*d1,pow_2_ndata):
        for j in range(n):
            m[i+j] = 1
    mn = heaan.Block(context,encrypted = False, data = m)
    
    gini = m0.encrypt()

    for i in range(n1):
        right = m0.encrypt()
        left = m0.encrypt()
        squareSum = m0.encrypt()

        for k in range(t):
            label = 'label_'+str(k+1)
            label_ctxt = label_ctxt.load(ctxt_path+L.id+'/'+label+'.ctxt')
            label_ctxt = scal * label_ctxt

            n_total = m0.encrypt()
            n_left = m0.encrypt()
            for j in range(n):
                name = 'X'+str(i+1)+'_'+str(j+1)
                X_ctxt = X_ctxt.load(ctxt_path+L.id+'/'+name+'.ctxt')
                ctmp = X_ctxt * label_ctxt
                
                r = 1
                while (r < pow_2_ndata):
                    rot = ctmp.__lshift__(r)
                    ctmp = rot + ctmp
                    r*=2
                ctmp = m_d1 * ctmp

                n_total = ctmp + n_total
                ctmp = ctmp.__rshift__(j)
                n_left = ctmp + n_left
            
            n_total = right_rotate_reduce(n_total,1,n)
            n_left = right_rotate_reduce(n_left,1,n)    
            n_right = n_total - n_left
                       
            left = n_left + left
            right = n_right + right
            n_left = n_left * n_left
            n_right = n_right * n_right
            squareSum = n_left + squareSum
            squareSum = n_right + squareSum

        left = left * left
        right = right * right
  
        left = right + left
        left = squareSum + left

        left = mn * left
        left = left.__rshift__(n*i)

        gini = left + gini 

    start = time.time()
    r = 1
    while(r<d1):
        rot = gini.__lshift__(pow_2_ndata-(n*n1))
        gini = rot + gini
        r*=2
    end = time.time()
    print(L.id, '로테이트로 값을 모아주기: %.5f sec' %(end-start))
    gini = mt * gini

    return gini

def DT_Learn_leaf(L,cdict,cy,d,n,t,ndata,model_path):
    pow_2_ndata = _get_smallest_pow_2(ndata)

    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))

    Lcy = m0.encrypt()
    
    start = time.time()
    for i in range(t):
        tmp = cy.__lshift__(i)
        tmp = tmp * m1
        tmp = (i+1) * tmp
        Lcy = tmp + Lcy
    L.cy = Lcy
    print(L.id, 'cy time: %.5f sec' %(time.time()-start))

    # cacc = m0.encrypt()

    # start = time.time()
    # cacc = left_rotate_reduce(cy,1,t)
    # cacc = m1 * cacc

    # L.by = cacc

    value = m0.encrypt()
    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = cdict[label]
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = tmp + rot
            r*=2
        value = tmp + value
    tmp = value * m1

    inverse_start = time.time()
    sample_inverse = tmp.inverse(greater_than_one = False)
    print(L.id, 'sample inverse time: %.5f sec' %(time.time()-inverse_start))
    sample_inverse = tmp * sample_inverse
    L.by = sample_inverse
    print(L.id, 'by time: %.5f sec' %(time.time()-start))

    print('cy: ')
    print_ctxt(L.cy,t)
    print('by: ')
    print_ctxt(L.by,1)
    
    return L.cy,L.by

def DT_Learn_leaf_big(L,cy,d,n,t,ndata,model_path,ctxt_path):
    pow_2_ndata = _get_smallest_pow_2(ndata)
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    mt = heaan.Block(context,encrypted = False, data = [1/(t*pow_2_ndata)]+[0]*(num_slot-1))
    empty_msg = heaan.Block(context,encrypted = False)
    
    Lcy = m0.encrypt()
    start = time.time() # evaluation으로 넘겨주기 위한 cy
    for i in range(t):
        tmp = cy.__lshift__(i)
        tmp = tmp * m1
        tmp = (i+1) * tmp
        Lcy = tmp + Lcy
    L.cy = Lcy
    print(L.id, 'cy time: %.5f sec' %(time.time()-start))

    # L.cy.to_device()
    # # print('DT_Learn 2', flush = True)
    # cacc = heaan.Ciphertext(context)
    # enc.encrypt(m0,kpack,cacc)
    # cacc.to_device()

    # start = time.time()
    # eval.left_rotate_reduce(cy,1,t,cacc)
    # mult(cacc,m1,cacc,eval)

    value = m0.encrypt()
    tmp = empty_msg.encrypt()
    for i in range(1,t+1):
        label = 'label_'+str(i)
        tmp = tmp.load(ctxt_path+L.id+'/'+label+'.ctxt')
        r = 1
        while (r<pow_2_ndata):
            rot = tmp.__lshift__(r)
            tmp = tmp + rot
            r*=2
        value = tmp + value
    tmp = value * mt

    inverse_start = time.time()
    sample_inverse = tmp.inverse(greater_than_one = False)
    print(L.id, 'sample inverse time: %.5f sec' %(time.time()-inverse_start))
    sample_inverse = tmp * sample_inverse
    sample_inverse = sample_inverse * sample_inverse
    L.by = sample_inverse
    print(L.id, 'by time: %.5f sec' %(time.time()-start))

    print('cy: ', flush = True)
    print_ctxt(L.cy,1)
    print('by: ', flush = True)
    print_ctxt(L.by,1)  
    
    return L.cy,L.by

def Multcacv(ca,cv,interval,num_node,d,n,t):
    
    m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)
    m1 = heaan.Block(context,encrypted = False, data = [1]+[0]*(num_slot-1))
    
    m0 = m0.encrypt()
    res_cv = m0.encrypt()
    for i in range(num_node):
        rot_ca = m0.encrypt()
        rot_cv = m0.encrypt()
        itv = ca.__lshift__(interval*i)
        itv2 = cv.__lshift__(interval*i)
        for j in range(d):
            rot = itv.__lshift__(j)
            rot = m1 * rot
            rot = right_rotate_reduce(rot,1,n)
            rot2 = itv2 * rot
            rot = rot.__rshift__(n*j)
            rot2 = rot2.__rshift__(n*j)
            rot_ca = rot_ca + rot
            rot_cv = rot2 + rot_cv

        rot = rot_ca.__rshift__(interval*i)
        rot2 = rot_cv.__lshift__(interval*i)
        res_ca = rot + res_ca
        res_cv = res_cv + rot2
 
    jina = res_ca * res_cv

    return jina

def UpdateCy(model_path,node_key,cy,by,interval,num_node,num_ctxt,parent_ctxt_num,d,n,t,depth):
    # num node : 해당 암호문에 들어간 노드 수
    # num_ctxt : 몇번째 암호문인지에 대한 정보
    m = [0]*num_slot
    for i in range(num_node[num_ctxt]):
        m[interval*i] = 1
    m1 = heaan.Block(context,encrypted = False, data = m)
    ctxt1 = m1.encrypt()

    empty_msg = heaan.Block(context,encrypted = False)
    parent_cy = empty_msg.encrypt() 
    parent_level = node_key-2
    if len(num_node) == parent_ctxt_num:
        parent_cy = parent_cy.load(model_path+"d"+str(parent_level)+"_tmpCy"+str(num_ctxt)+".ctxt")
    else:
        if parent_ctxt_num == 1:
            parent_cy = parent_cy.load(model_path+"d"+str(parent_level)+"_tmpCy0.ctxt")
        else:
            parent_cy = parent_cy.load(model_path+"d"+str(parent_level)+"_tmpCy"+str(int((num_ctxt//parent_ctxt_num)))+".ctxt")
    
    # 1-by
    minus_by = ctxt1 - by
    
    # tmpCy = cy*by + parent_cy*(1-by)
    ctmp1 = cy * by
    ctmp2 = parent_cy * minus_by

    tmpCy = ctmp1 + ctmp2

    return tmpCy

def preCom(save_path,model_path,node_key,ca,cv,cy,by,num_list,interval,parent_ctxt_num,d,n,t,depth):
    
    for i in range(len(num_list)):
        if node_key != depth+1:
            cacv = Multcacv(ca[i],cv[i],interval,num_list[i],d,n,t)
            cacv.save(save_path+"d"+str(node_key-1)+"_cacv"+str(i)+".ctxt")

        if node_key != 1:
            tmpCy = UpdateCy(model_path,node_key,cy[i],by[i],interval,num_list,i,parent_ctxt_num,d,n,t,depth)
            tmpCy.save(model_path+"d"+str(node_key-1)+"_tmpCy"+str(i)+".ctxt")
        else:
            cy[i].save(model_path+"d"+str(node_key-1)+"_tmpCy0.ctxt") ## root node
        
    return

###############################################################
####################   random forest  #########################
###############################################################
def make_random_data(ndata, pow_2_ndata, d, n, logN):
    # 2번 뽑힐 데이터=2, 1번 뽑히는 데이터=1 을 넣은 벡터 만들기
    rand_ = random.choices(range(ndata), k=round(ndata*0.632))
    twice_num = round(ndata*0.1896)
    once_num = round(ndata*0.4424)
    twice_list = rand_[0:twice_num]
    once_list = rand_[twice_num:]

    sampling_ind = np.array([0.0]*(1<<(logN-1)))
    sampling_csel = np.array([0.0]*(1<<(logN-1)))
    for i in range(round(ndata*0.632)):
        if i < round(ndata*0.1896):
            sampling_ind[rand_[i]] = 2
            sampling_csel[rand_[i]] = 0.5
        else:
            sampling_ind[rand_[i]] = 1
            sampling_csel[rand_[i]] = 1

    tmp1 = np.array([0.0]*(1<<(logN-1)))
    tmp2 = np.array([0.0]*(1<<(logN-1)))
    # right_rotate_reduce(sel, N, n, tmp_sel)
    for i in range(n):
        if i>0:
            sampling_ind = np.roll(sampling_ind, pow_2_ndata)
        tmp1 = sampling_ind + tmp1
    # right_rotate_reduce(tmp_sel, N*pow2_n, d_in_ctxt, tmp_sel)
    for i in range(d):
        if i>0:
            tmp1 = np.roll(tmp1, pow_2_ndata*n)
        tmp2 = tmp1 + tmp2
        
    return tmp2, sampling_csel

def random_feature(d,logN):
    feature_list = random.sample(range(d), round(math.sqrt(d)))
    m1_ = [0]*(1<<(logN-1)) # 선택된 변수
    for i in feature_list:
        m1_[i] = 1
    m2_ = np.array([1]*(1<<(logN-1))) - np.array(m1_) # 선택되지 않은 변수
    # print(m1_, m2_)
    m1 = m1_
    m2 = m2_

    # 메세지 형태로 리턴, 한 번 곱하고(m1) 한 번 더하고(m2) 안 쓸거임
    return m1, m2
