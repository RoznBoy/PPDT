import piheaan as heaan
import pandas as pd
import numpy as np
import warnings
from IPython.display import display
import math
import json
import os
import natsort
import re

# name_list = []
# def node_list(name, depth):
#     name_list.append(name)
#     if len(name) == depth+1:
#         return
    
#     new = name+'l'
#     node_list(new, depth)
#     new = name+'r'
#     node_list(new, depth)
    
#     return name_list
    

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
        df[cname].values.categories = tmp_cat

    df = df.astype('int64')
    
    ndata = df.shape[0]
    d = len(col)-1
    n = find_max_cat_X(df)
    t = len(df['label'].unique())

    return ndata, d, n, t

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


def check_boot(ctxt, eval_):
    if ctxt.level -3 < eval_.min_level_for_bootstrap:
        eval_.bootstrap(ctxt, ctxt)
    return ctxt

def Bc(ctxt,context,kpack,logN,d,n):
    he = heaan.HomEvaluator(context,kpack)
    one_0 = heaan.Message(logN-1,0)
    one_0[0]=1.0
    s=1
    c = heaan.Ciphertext(ctxt)
    cn = heaan.Ciphertext(context)
    while (s<(2 << (logN-2))):
        he.left_rotate(c,s,cn)
        he.add(c,cn,c)
        s*=2
    
    he.mult(c,one_0,c)
    return c

def approxDiscreteEqualZero(ev,kpack,cin,context,sk):
    if (not kpack.is_mult_key_loaded()): kpack.load_mult_key()
    if (not kpack.is_conj_key_loaded()): kpack.load_conj_key()
    dec = heaan.Decryptor(context)
    scale_factor = 21 ## makes 1 when less than 2^{-15.8} 
    degree = 4
    depth_cost = 9
    n=4
    d=10
    logN=16

    sinc_coeffs=[1.0,0,-1.0/6,0,1.0/120]
    cos_coeffs=[1,0,-1/2,0,1/24]
    
    cx = heaan.Ciphertext(cin)
    check_boot(cx, ev)
    ##print("approx input:cx")
    ##print_ctxt(cx,dec,sk,logN,n*d)

    ev.mult(cx,201/4096,cx)

    c_pows={}
    c_pows[1]=cx
    for i in range(2,degree+1):
        if (i % 2 == 0):
            c_pows[i] = heaan.Ciphertext(context)
            ev.square(c_pows[i//2],c_pows[i])
            check_boot(c_pows[i], ev)
        else:
            c_pows[i] = heaan.Ciphertext(context)
            ev.mult(c_pows[i//2],c_pows[i//2+1],c_pows[i])
            check_boot(c_pows[i], ev)

    ctxt_tmp = heaan.Ciphertext(context)
    ## compute sinc(x/2^d)
    for i in range(1,degree+1):
        ev.mult(c_pows[i],sinc_coeffs[i],ctxt_tmp)
        check_boot(ctxt_tmp, ev)
        if i==1:
            c_sinc = heaan.Ciphertext(ctxt_tmp)
        else:
            ev.add(c_sinc,ctxt_tmp,c_sinc)

    ev.add(c_sinc,sinc_coeffs[0],c_sinc)


    ## compute cos(x/2^d)
    for i in range(1,degree+1):
        ev.mult(c_pows[i],cos_coeffs[i],ctxt_tmp)
        if i==1:
            c_cos = heaan.Ciphertext(ctxt_tmp)
        else:
            ev.add(c_cos,ctxt_tmp,c_cos)

    ev.add(c_cos,cos_coeffs[0],c_cos)

    check_boot(c_cos, ev)
    check_boot(c_sinc, ev)

    for i in range(scale_factor):
        ## compute sinc(2x)
        ev.mult(c_sinc,c_cos,c_sinc)
        
        ## compute cos(2x)
        ev.square(c_cos,c_cos)
        ev.mult(c_cos,2,c_cos)
        ev.add(c_cos,-1,c_cos)
        check_boot(c_cos, ev)
        check_boot(c_sinc, ev)

    ## applying filter
    cout=c_sinc 
    ev.sub(cout,0.5,cout)
    ##print("cout.level(s):",cout.level)
    ##print_ctxt(cout,dec,sk,logN,n*d)
    cout=approxSign(ev,kpack,cout,8,3,context)
    ev.add(cout,1,cout)
    ev.mult(cout,0.5,cout)
    check_boot(cout, ev)
    ##print("cout.level:",cout.level)
    ##print_ctxt(cout,dec,sk,logN,n*d)
    return cout

def findMax4(c, context, kpack, logN, d, n, ndata,dec,sk):
    if (ndata==1): 
        print("findMax ends..")
        print_ctxt(c,dec,sk,logN-1,1)
        return c

    he = heaan.HomEvaluator(context,kpack)
    ##print("cinx level:",c.level)
    check_boot(c,he)
    ## divide ctxt by four chunks
    if (ndata % 4 !=0):
        i=ndata
        msg = heaan.Message(logN-1,0)
        while (i % 4 !=0):
            msg[i]=0.00000
            i+=1
        ndata=i
        he.add(c,msg,c)

    print("Max: after making 4")
    print_ctxt(c,dec,sk,logN-1,ndata)
    ## Divide c by 4 chunk
    msg1=heaan.Message(logN-1)
    for i in range(ndata//4):
        msg1[i]=1
    ca = heaan.Ciphertext(context)
    cb = heaan.Ciphertext(context)
    cc = heaan.Ciphertext(context)
    cd = heaan.Ciphertext(context)
    c1 = heaan.Ciphertext(context)
    c2 = heaan.Ciphertext(context)
    c3 = heaan.Ciphertext(context)
    c4 = heaan.Ciphertext(context)
    c5 = heaan.Ciphertext(context)
    c6 = heaan.Ciphertext(context)
    ctmp1 = heaan.Ciphertext(context)
    ctmp2 = heaan.Ciphertext(context)

    he.mult(c,msg1,ca)
    he.left_rotate(c,ndata//4,ctmp1)
    he.mult(ctmp1,msg1,cb)
    he.left_rotate(c,ndata//2,ctmp1)
    he.mult(ctmp1,msg1,cc)
    he.left_rotate(c,ndata*3//4,ctmp1)
    he.mult(ctmp1,msg1,cd)

    check_boot(ca,he)
    check_boot(cb,he)
    check_boot(cc,he)
    check_boot(cd,he)

    he.sub(ca,cb,c1)
    he.sub(cb,cc,c2)
    he.sub(cc,cd,c3)
    he.sub(cd,ca,c4)
    he.sub(ca,cc,c5)
    he.sub(cb,cd,c6)

    he.right_rotate(c2,ndata//4,ctmp1)
    he.add(ctmp1,c1,ctmp1)

    he.right_rotate(c3,ndata//2,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)

    he.right_rotate(c4,ndata*3//4,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)

    he.right_rotate(c5,ndata,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)

    he.right_rotate(c6,5*ndata//4,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)
    ## 
    ##print("approxSign input")
    ##print_ctxt(ctmp1,dec,sk,17,d*n)

    c0=approxSign(he,kpack,ctmp1, 8, 3,context)
    c0_c = heaan.Ciphertext(c0)
    print("Max: c0_c")
    print_ctxt(c0_c,dec,sk,logN-1,ndata*3//2)

    mkall = heaan.Message(logN-1,1.0)
    he.add(c0,mkall,c0)
    he.mult(c0,0.5,c0)
    check_boot(c0,he)



    ## Making equality ciphertext 
    ceq = heaan.Ciphertext(context)
    he.square(c0_c,ceq)
    check_boot(ceq,he)
    he.negate(ceq,ceq)
    he.add(ceq,mkall,ceq)

    #22print("ceq: ")
    #22print_ctxt(ceq,dec,sk,logN-1,ndata*3//2)


    ## Step 6..
    mk1=msg1
    mk2=heaan.Message(logN-1)
    mk3=heaan.Message(logN-1)
    mk4=heaan.Message(logN-1)
    mk5=heaan.Message(logN-1)
    mk6=heaan.Message(logN-1)

    he.right_rotate(mk1,ndata//4,mk2)
    he.right_rotate(mk2,ndata//4,mk3)
    he.right_rotate(mk3,ndata//4,mk4)
    he.right_rotate(mk4,ndata//4,mk5)
    he.right_rotate(mk5,ndata//4,mk6)

    ## Step 7
    c_neg = heaan.Ciphertext(c0)
    he.negate(c0,c_neg)

    he.add(c_neg,mkall,c_neg) ## c_neg = 1-c0 

    ### When min=a
    c0n = heaan.Ciphertext(c0)
    he.mult(c0n,mk1,ctmp1)
    ctxt=c0n
    ##he.mult(c0,mk1,ctmp1)
    c_ab = heaan.Ciphertext(ctmp1)
    
    ## ctmp2 = a>d
    c0=heaan.Ciphertext(ctxt)
    he.mult(c_neg,mk4,ctmp2)
    he.left_rotate(ctmp2,(3*ndata)//4,ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)

    ## ctmp2 = a>c
    he.mult(c0,mk5,ctmp2)
    he.left_rotate(ctmp2,ndata,ctmp2)

    ## cca
    cca = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,cca)

    ## Min=b
    he.mult(c_neg,mk1,ctmp1)
    he.mult(c0,mk2,ctmp2)
    he.left_rotate(ctmp2,ndata//4,ctmp2)
    c_bc = heaan.Ciphertext(ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)

    he.mult(c0,mk6,ctmp2)
    he.left_rotate(ctmp2,ndata*5//4,ctmp2)
    ccb = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccb)

    ## Min=c
    he.mult(c_neg,mk2,ctmp1)
    he.left_rotate(ctmp1,ndata//4,ctmp1)
    he.mult(c0,mk3,ctmp2)
    he.left_rotate(ctmp2,ndata//2,ctmp2)
    c_cd = heaan.Ciphertext(ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)
    he.mult(c_neg,mk5,ctmp2)
    he.left_rotate(ctmp2,ndata,ctmp2)
    ccc = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccc)

    ## Min=d
    he.mult(c_neg,mk3,ctmp1)
    he.left_rotate(ctmp1,ndata//2,ctmp1)
    he.mult(c0,mk4,ctmp2)
    he.left_rotate(ctmp2,3*ndata//4,ctmp2)
    cda = heaan.Ciphertext(ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)

    he.mult(c_neg,mk6,ctmp2)
    he.left_rotate(ctmp2,5*ndata//4,ctmp2)

    ccd = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccd)

    check_boot(cca,he)
    check_boot(ccb,he)
    check_boot(ccc,he)
    check_boot(ccd,he)

    he.mult(cca,ca,cca)
    he.mult(ccb,cb,ccb)
    he.mult(ccc,cc,ccc)
    he.mult(ccd,cd,ccd)
    
    cout = heaan.Ciphertext(cca)
    he.add(cout,ccb,cout)
    he.add(cout,ccc,cout)
    he.add(cout,ccd,cout)

    check_boot(cout,he)

    ### Going to check equality
    cneq = heaan.Ciphertext(context)
    he.negate(ceq,cneq)
    he.add(cneq,mkall,cneq) ## cneq = 1-ceq
    cneq_da = heaan.Ciphertext(context)
    he.left_rotate(cneq,(3*ndata)//4,cneq_da)
    he.mult(cneq_da,mk1,cneq_da)

    cneq_bc = heaan.Ciphertext(context)
    he.left_rotate(cneq,(ndata)//4,cneq_bc)
    he.mult(cneq_bc,mk1,cneq_bc)

    ceq_ab = heaan.Ciphertext(context)
    he.mult(ceq,mk1,ceq_ab)

    ceq_bc = heaan.Ciphertext(context)
    he.left_rotate(ceq,(ndata)//4,ceq_bc)
    he.mult(ceq_bc,mk1,ceq_bc)


    ceq_cd = heaan.Ciphertext(context)
    he.left_rotate(ceq,(ndata)//2,ceq_cd)
    he.mult(ceq_cd,mk1,ceq_cd)

    ceq_da = heaan.Ciphertext(context)
    he.negate(cneq_da,ceq_da)
    he.add(ceq_da,mk1,ceq_da)

    ## Checking remaining depth
    check_boot(ceq,he)
    check_boot(ceq_ab,he)
    check_boot(ceq_bc,he)
    check_boot(ceq_cd,he)
    check_boot(ceq_da,he)

    ## Need to check (a=b)(b=c)(c>d)
    he.mult(ceq_ab,ceq_bc,ctmp2)
    he.mult(ctmp2,c_cd,ctmp1)
    check_boot(ctmp1, he)
    c_cond3 = heaan.Ciphertext(ctmp1)

    ## (b=c)(c=d)(d>a)
    he.mult(ceq_bc,ceq_cd,ctmp1)
    he.mult(ctmp1,cda,ctmp1)
    he.add(c_cond3,ctmp1,c_cond3)

    ## (c=d)(d=a)(a>b)
    he.mult(ceq_cd,ceq_da,ctmp1)
    he.mult(ctmp1,c_ab,ctmp1)
    he.add(c_cond3,ctmp1,c_cond3)

    ## (a=b)(d=a)(b>c)
    he.mult(ceq_ab,ceq_da,ctmp1)
    he.mult(ctmp1,c_bc,ctmp1)
    he.add(c_cond3,ctmp1,c_cond3)

    c_cond4 = heaan.Ciphertext(context)
    he.mult(ctmp2,ceq_cd,c_cond4)

    check_boot(c_cond3, he)
    check_boot(c_cond4, he)

    c_tba = heaan.Ciphertext(context)
    he.mult(c_cond3,0.333333333,c_tba)
    check_boot(c_tba, he)
    he.add(c_tba,mkall,c_tba)
    check_boot(c_cond4, he)
    he.add(c_cond4,mkall,ctmp1)   
    he.mult(c_tba,ctmp1,c_tba)
    he.mult(cout,c_tba,cout)
    check_boot(cout, he)

    return findMax4(cout, context, kpack, logN, d, n, ndata//4,dec,sk)

def findMaxPos(c,context,kpack,logN,d,n,ndata,dec,sk):
    cmax = findMax4(c,context,kpack,logN,d,n,ndata,dec,sk)
    ## Extension
    ## Suppose slot size of 2^N form
    he = heaan.HomEvaluator(context,kpack)
    ctmp = heaan.Ciphertext(context)
    for i in range(logN-1):
        he.right_rotate(cmax,pow(2,i),ctmp)
        he.add(cmax,ctmp,cmax)
    he.sub(c,cmax,c)
    #22print('c-cmax: ')
    #22print_ctxt(c,dec,sk,logN,ndata)
    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    c_red = approxDiscreteEqualZero(he,kpack,c,context,sk)
    print("c_red.level:",c_red.level)
    print_ctxt(c_red,dec,sk,logN,ndata)

    ### Need to generate a rotate sequence to choose 1 in a random position   
    c_out=selectRandomOnePos(c_red,context,kpack,logN,d,n,ndata,dec,sk)
    print("selRandOne cmax:",c_out.level)
    print_ctxt(c_out,dec,sk,logN,ndata)
    return c_out

def selectRandomOnePos(c_red,context,kpack,logN,d,n,ndata,dec,sk):
    eval_ = heaan.HomEvaluator(context,kpack)
    enc = heaan.Encryptor(context)

    m0 = heaan.Message(logN-1,0.0)
    c_sel = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,c_sel)
    rando = np.random.permutation(ndata)
    ctmp1 = heaan.Ciphertext(c_red)
    check_boot(ctmp1, eval_)
    ctmp2 = heaan.Ciphertext(context)
    m0[0]=1.0
    for l in rando:
        if (l>0):
            check_boot(ctmp1, eval_)
            check_boot(c_sel, eval_)

            eval_.left_rotate(ctmp1,l,ctmp1)
            eval_.mult(c_sel,ctmp1,ctmp2)
            eval_.sub(ctmp1,ctmp2,ctmp1)
            eval_.mult(ctmp1,m0,ctmp2)
            eval_.right_rotate(ctmp1,l,ctmp1)
            eval_.add(c_sel,ctmp2,c_sel)
        else:
            check_boot(ctmp1, eval_)
            check_boot(c_sel, eval_)
            eval_.mult(c_sel,ctmp1,ctmp2)
            eval_.sub(ctmp1,ctmp2,ctmp1)
            eval_.mult(ctmp1,m0,ctmp2)
            eval_.add(c_sel,ctmp2,c_sel)

    check_boot(ctmp1, eval_)

    return ctmp1

def findMinPos(c,context,kpack,logN,d,n,ndata,dec,sk):
    dec = heaan.Decryptor(context)
    cmin = findMin4(c,context,kpack,logN,d,n,ndata,dec,sk)
    print("findMinPos cmin:",cmin.level)
    print_ctxt(cmin,dec,sk,logN,1)


    ## Extension
    ## Suppose slot size of 2^N form
    he = heaan.HomEvaluator(context,kpack)
    ctmp = heaan.Ciphertext(context)
    for i in range(logN-1):
        he.right_rotate(cmin,pow(2,i),ctmp)
        he.add(cmin,ctmp,cmin)
    he.sub(c,cmin,c)
    
    ## Need to add a routine to check if the result of approxDEZ have all 1s --> No y value exists
    ## Possible that c_red may have many 1s. We need to select one (in current situation 20220810)
    c_red = approxDiscreteEqualZero(he,kpack,c,context,sk)
    print("c_red.level:",c_red.level)


    ### Need to generate a rotate sequence to choose 1 in a random position   
    c_out=selectRandomOnePos(c_red,context,kpack,logN,d,n,ndata,dec,sk)
    print("selRandOne cmin:",c_out.level)
    print_ctxt(c_out,dec,sk,logN,n*d)
    return c_out

def findMin4(c, context, kpack, logN, d, n, ndata,dec,sk):
    if (ndata==1): 
        print("findMin4 ends..")
        print_ctxt(c,dec,sk,logN,1)
        return c

    he = heaan.HomEvaluator(context,kpack)
    ##print("cinn level:",c.level)
    check_boot(c, he)
    ## divide ctxt by four chunks
    if (ndata % 4 !=0):
        i=ndata
        msg = heaan.Message(logN-1,0)
        while (i % 4 !=0):
            msg[i]=1.0000
            i+=1
        ndata=i
        he.add(c,msg,c)
    ##print("size: ",ndata)
    ##print_ctxt(c, dec, sk, logN, ndata)

    ## Divide c by 4 chunk
    msg1=heaan.Message(logN-1)
    for i in range(ndata//4):
        msg1[i]=1
    ca = heaan.Ciphertext(context)
    cb = heaan.Ciphertext(context)
    cc = heaan.Ciphertext(context)
    cd = heaan.Ciphertext(context)
    c1 = heaan.Ciphertext(context)
    c2 = heaan.Ciphertext(context)
    c3 = heaan.Ciphertext(context)
    c4 = heaan.Ciphertext(context)
    c5 = heaan.Ciphertext(context)
    c6 = heaan.Ciphertext(context)
    ctmp1 = heaan.Ciphertext(context)
    ctmp2 = heaan.Ciphertext(context)

    he.mult(c,msg1,ca)
    he.left_rotate(c,ndata//4,ctmp1)
    he.mult(ctmp1,msg1,cb)
    he.left_rotate(c,ndata//2,ctmp1)
    he.mult(ctmp1,msg1,cc)
    he.left_rotate(c,ndata*3//4,ctmp1)
    he.mult(ctmp1,msg1,cd)
    check_boot(ca, he)
    check_boot(cb, he)
    check_boot(cc, he)
    check_boot(cd, he)
    he.sub(ca,cb,c1)
    he.sub(cb,cc,c2)
    he.sub(cc,cd,c3)
    he.sub(cd,ca,c4)
    he.sub(ca,cc,c5)
    he.sub(cb,cd,c6)

    he.right_rotate(c2,ndata//4,ctmp1)
    he.add(ctmp1,c1,ctmp1)
    check_boot(ctmp1, he)

    he.right_rotate(c3,ndata//2,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)

    he.right_rotate(c4,ndata*3//4,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)

    he.right_rotate(c5,ndata,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)
    
    he.right_rotate(c6,5*ndata//4,ctmp2)
    he.add(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)
    
    c0=approxSign(he,kpack,ctmp1, 8, 3,context)
    c0_c = heaan.Ciphertext(c0)
    mkall = heaan.Message(logN-1,1.0)
    he.add(c0,mkall,c0)
    he.mult(c0,0.5,c0)
    check_boot(c0, he)
    ## Making equality ciphertext 
    ceq = heaan.Ciphertext(context)
    he.square(c0_c,ceq)
    check_boot(ceq, he)
    he.negate(ceq,ceq)
    he.add(ceq,mkall,ceq)



    ## Step 6..
    mk1=msg1
    mk2=heaan.Message(logN-1)
    mk3=heaan.Message(logN-1)
    mk4=heaan.Message(logN-1)
    mk5=heaan.Message(logN-1)
    mk6=heaan.Message(logN-1)

    he.right_rotate(mk1,ndata//4,mk2)
    he.right_rotate(mk2,ndata//4,mk3)
    he.right_rotate(mk3,ndata//4,mk4)
    he.right_rotate(mk4,ndata//4,mk5)
    he.right_rotate(mk5,ndata//4,mk6)

    ## Step 7
    c_neg = heaan.Ciphertext(c0)
    he.negate(c0,c_neg)

    he.add(c_neg,mkall,c_neg) ## c_neg = 1-c0 

    ### When min=a
    c0n = heaan.Ciphertext(c0)
    he.mult(c_neg,mk1,ctmp1)
    check_boot(ctmp1, he)
    ctxt=c0n
    ##he.mult(c0,mk1,ctmp1)
    
    c0=heaan.Ciphertext(ctxt)
    he.mult(c0,mk4,ctmp2)
    check_boot(ctmp2, he)
    ##he.mult(c_neg,mk4,ctmp2)
    he.left_rotate(ctmp2,(3*ndata)//4,ctmp2)
    cda = heaan.Ciphertext(ctmp2) ## (d>a) 
    he.mult(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)

    he.mult(c_neg,mk5,ctmp2)
    check_boot(ctmp2, he)
    #he.mult(c0,mk5,ctmp2)
    he.left_rotate(ctmp2,ndata,ctmp2)
    ## cca
    cca = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,cca)
    check_boot(cca, he)

    ## Min=b
    he.mult(c0,mk1,ctmp1)
    check_boot(ctmp1, he)
    he.mult(c_neg,mk2,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,ndata//4,ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)

    he.mult(c_neg,mk6,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,ndata*5//4,ctmp2)
    ccb = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccb)
    check_boot(ccb, he)

    ## Min=c
    he.mult(c0,mk2,ctmp1)
    check_boot(ctmp1, he)
    he.left_rotate(ctmp1,ndata//4,ctmp1)
    cbc = heaan.Ciphertext(ctmp1) ## (b>c)

    he.mult(c_neg,mk3,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,ndata//2,ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)
    he.mult(c0,mk5,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,ndata,ctmp2)
    ccc = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccc)
    check_boot(ccc, he)

    ## Min=d
    he.mult(c0,mk3,ctmp1)
    check_boot(ctmp1, he)
    he.left_rotate(ctmp1,ndata//2,ctmp1)
    he.mult(c_neg,mk4,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,3*ndata//4,ctmp2)
    he.mult(ctmp1,ctmp2,ctmp1)
    check_boot(ctmp1, he)

    he.mult(c0,mk6,ctmp2)
    check_boot(ctmp2, he)
    he.left_rotate(ctmp2,5*ndata//4,ctmp2)

    ccd = heaan.Ciphertext(context)
    he.mult(ctmp1,ctmp2,ccd)
    check_boot(cca, he)
    check_boot(ccb, he)
    check_boot(ccc, he)
    check_boot(ccd, he)

    he.mult(cca,ca,cca)
    he.mult(ccb,cb,ccb)
    he.mult(ccc,cc,ccc)
    he.mult(ccd,cd,ccd)
    check_boot(cca, he)
    check_boot(ccb, he)
    check_boot(ccc, he)
    check_boot(ccd, he)
    
    cout = heaan.Ciphertext(cca)
    he.add(cout,ccb,cout)
    he.add(cout,ccc,cout)
    he.add(cout,ccd,cout)

    check_boot(cout, he)

    ### Going to check equality
    ceq_ab = heaan.Ciphertext(context)
    he.mult(ceq,mk1,ceq_ab)

    ceq_bc = heaan.Ciphertext(context)
    he.left_rotate(ceq,(ndata)//4,ceq_bc) 
    he.mult(ceq_bc,mk1,ceq_bc)


    ceq_cd = heaan.Ciphertext(context)
    he.left_rotate(ceq,(ndata)//2,ceq_cd) 
    he.mult(ceq_cd,mk1,ceq_cd)

    ceq_da = heaan.Ciphertext(context)
    he.left_rotate(ceq,(ndata)*3//4,ceq_da) 
    he.mult(ceq_da,mk1,ceq_da)



    ## Checking remaining depth
    check_boot(ceq, he)
    check_boot(ceq_ab, he)
    check_boot(ceq_bc, he)
    check_boot(ceq_cd, he)
    check_boot(ceq_da, he)
    check_boot(cbc, he)
    check_boot(cda, he)

    ncda = heaan.Ciphertext(cda)
    he.negate(ncda,ncda)
    he.add(ncda,mk1,ncda)

    ncbc = heaan.Ciphertext(cbc)
    he.negate(ncbc,ncbc)
    he.add(ncbc,mk1,ncbc)
    ##print("d=a")
    ##print_ctxt(ceq_da,dec,sk,logN,ndata)
    ## (a=b)(b=c)(d>a)
    he.mult(ceq_ab,ceq_bc,ctmp2)
    he.mult(ctmp2,cda,ctmp1)
    c_cond3 = heaan.Ciphertext(ctmp1)
    ## (b=c)(c=d)(1-(d>a)) 
    he.mult(ceq_bc,ceq_cd,ctmp1)
    he.mult(ctmp1,ncda,ctmp1)
    check_boot(ctmp1, he)
    he.add(c_cond3,ctmp1,c_cond3)
    ## (c=d)(d=a)(b>c)
    he.mult(ceq_cd,ceq_da,ctmp1)
    he.mult(ctmp1,cbc,ctmp1)
    check_boot(ctmp1, he)
    he.add(c_cond3,ctmp1,c_cond3)
    ## (d=a)(a=b)(1-(b>c))
    he.mult(ceq_ab,ceq_da,ctmp1)
    he.mult(ctmp1,ncbc,ctmp1)
    check_boot(ctmp1, he)
    he.add(c_cond3,ctmp1,c_cond3)
  
    c_cond4 = heaan.Ciphertext(context)
    he.mult(ctmp2,ceq_cd,c_cond4)
    check_boot(c_cond4, he)
    """
    print("c_cond3:")
    print_ctxt(c_cond3,dec,sk,logN,ndata)
    print("c_cond4:")
    print_ctxt(c_cond4,dec,sk,logN,ndata)
    """
    c_tba = heaan.Ciphertext(context)
    he.mult(c_cond3,0.333333333,c_tba)
    he.add(c_tba,mkall,c_tba)
    he.mult(c_cond4,0.333333333,c_cond4)
    he.add(c_cond4,c_tba,c_tba)
    he.mult(cout,c_tba,cout)
    
    check_boot(cout,he)
    return findMin4(cout, context, kpack, logN, d, n, ndata//4,dec,sk)

## listX: list of X variables, listY: list of Y variable
## 
def findMax2IG(cdict,listX,listY):
    return None

def no_zero_value(df, col):
    one = [1] * len(col)
    plus_one = []
    for i in range(df.shape[0]):
        plus_one.append(one)
        np.array(plus_one)
    df = df.to_numpy()
    df = pd.DataFrame(plus_one + df)
    return df

def find_max_cat_X(df):
    col = df.columns
    max_values = []
    for i in col.drop('label'):
        max_values.append(max(df[i]))
    max_x = int(max(max_values))
    return max_x

# def get_dummies_max(df, col, max_x, max_values):
#     df_dummy = pd.get_dummies(df, columns=col)
#     idx =[]
#     for c in df_dummy.columns:
#         if 'label' in c:
#             idx.append(c)
#     for j in range(1,len(max_values)+1):
#         for k in range(1, max_x+1):
#             idx.append(f'X{j}_{k}')
#             test = df_dummy.reindex(idx, axis=1, fill_value = 0)
#     return test

# def convert_with_binning(name1):
#     warnings.filterwarnings('ignore')
#     df = pd.read_csv(name1)
#     col = df.columns

#     df = no_zero_value(df)
#     max_x = find_max_cat_X(df,col)
#     df_dummy = get_dummies_max(df, col, max_x, max_values)
#     output_name = name1+'_p'
#     df_dummy.to_csv(output_name,mode='w')

def encrypt_and_save(df, ctxt_path, context, keypack):
    enc = heaan.Encryptor(context)
    ## obtain each column of df and encrypt
    for cname in df.columns:
        #print(cname)
        #print(type(df[cname]))
        #print(df[cname])
        msg = heaan.Message(15,0)
        for index in range(df[cname].size):
            msg[index]=df[cname][index]
   
        if ("Unnamed" not in cname):
            c = heaan.Ciphertext(context, 15)
            enc.encrypt(msg,keypack,c)
            c.save(ctxt_path+cname+".ctxt")
            
def enc_and_save(df, n, d, ndata, logN, ctxt_path, context, keypack):
    enc = heaan.Encryptor(context)
    ## obtain each column of df and encrypt
    pow_2 = _get_smallest_pow_2(ndata)
    x_msg = heaan.Message(logN-1)
    num = 0
    for cname in df.columns:
        if ('label' in cname):
            msg = heaan.Message(logN-1)
            for nd in range(n*d):
                for size in range(df[cname].size):
                    index = nd*pow_2+size
                    msg[index]=df[cname][size]
            label_ctxt = heaan.Ciphertext(context)
            enc.encrypt(msg,keypack,label_ctxt)
            label_ctxt.save(ctxt_path+cname+".ctxt")
        
        if ('X' in cname):
            for size in range(df[cname].size):
                index = num*pow_2+size
                x_msg[index]=df[cname][size]
            num+=1
                    
    c = heaan.Ciphertext(context)
    enc.encrypt(x_msg,keypack,c)
    c.save(ctxt_path+"X.ctxt")
    
def make_one_ctxt(df, ctxt_path, t, ndata, logN, context, pk):
    enc = heaan.Encryptor(context)
    X_input = []
    for xname in df.columns[t:]:
        new_x = df[xname].to_list() + [0]*(_get_smallest_pow_2(ndata) - ndata)
        X_input =  X_input + new_x
    input_ctxt_ = heaan.Message(logN-1, 0)
    x_ctxt = heaan.Ciphertext(context)
    for index in range(len(X_input)):
        input_ctxt_[index] = X_input[index]
    enc.encrypt(input_ctxt_, pk, x_ctxt)
    x_ctxt.save(ctxt_path + "x_ctxt.ctxt")

def load_ctxt(fn_list,ctxt_path,context): ## returns the list of ciphertexts loaded
    out_cdict={}
    for cname in fn_list:
        ctxt = heaan.Ciphertext(context)
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

def approxSign(eval_, pack, ctxt_in, n1, n2,context):
    degree = 4
    coeffs=[315/128,-420/128,378/128,-180/128,35/128]
    coeffs_g=[5850/1024, -34974.0/1024, 97015/1024, -113492/1024, 46623/1024]
    ctxt_pows={}
    one_iter_cost = 6

    ctxt_sign = ctxt_in
    if (not pack.is_mult_key_loaded()): pack.load_mult_key()
    if (not pack.is_conj_key_loaded()): pack.load_conj_key()
    for i in range(n1):
        check_boot(ctxt_sign, eval_)
        ctxt_pows[0]=ctxt_sign
        ctxt_pows[1]=heaan.Ciphertext(context)
        eval_.square(ctxt_pows[0],ctxt_pows[1])
        check_boot(ctxt_pows[1], eval_)
        for j in range(2,degree+1):
            ctxt_pows[j]=heaan.Ciphertext(context)
            if (j % 2 == 0):
               eval_.square(ctxt_pows[j/2],ctxt_pows[j])
               check_boot(ctxt_pows[j], eval_)
            else:
               eval_.mult(ctxt_pows[j//2],ctxt_pows[j//2+1],ctxt_pows[j])
               check_boot(ctxt_pows[j], eval_)

        ctxt_tmp1 = heaan.Ciphertext(context)
        ctxt_tmp2 = heaan.Ciphertext(context)
        eval_.mult(ctxt_pows[1],coeffs_g[1],ctxt_tmp1)
        check_boot(ctxt_tmp1, eval_)
        for j in range(2,degree+1):
            eval_.mult(ctxt_pows[j],coeffs_g[j],ctxt_tmp2)
            check_boot(ctxt_tmp2, eval_)
            eval_.add(ctxt_tmp1, ctxt_tmp2, ctxt_tmp1)

        eval_.add(ctxt_tmp1,coeffs_g[0],ctxt_tmp1)
        eval_.mult(ctxt_tmp1,ctxt_pows[0],ctxt_sign)
        check_boot(ctxt_sign, eval_)
        eval_.kill_imag(ctxt_sign,ctxt_sign)


    for i in range(n2):
        check_boot(ctxt_sign, eval_)

        ctxt_pows[0]=ctxt_sign
        eval_.square(ctxt_pows[0],ctxt_pows[1])
        check_boot(ctxt_pows[1], eval_)
        for j in range(2,degree+1):
            if (j % 2 == 0):
                eval_.square(ctxt_pows[j//2],ctxt_pows[j])
                check_boot(ctxt_pows[j], eval_)
            else:
                eval_.mult(ctxt_pows[j//2],ctxt_pows[j//2+1],ctxt_pows[j])
                check_boot(ctxt_pows[j], eval_)

        ctxt_tmp1 = heaan.Ciphertext(context)
        ctxt_tmp2 = heaan.Ciphertext(context)
        eval_.mult(ctxt_pows[1],coeffs[1],ctxt_tmp1)
        check_boot(ctxt_tmp1, eval_)
        for j in range(2,degree+1):
            eval_.mult(ctxt_pows[j],coeffs[j],ctxt_tmp2)
            check_boot(ctxt_tmp2, eval_)
            eval_.add(ctxt_tmp1, ctxt_tmp2, ctxt_tmp1)

        eval_.add(ctxt_tmp1,coeffs[0],ctxt_tmp1)
        eval_.mult(ctxt_tmp1,ctxt_pows[0],ctxt_sign)
        check_boot(ctxt_sign, eval_)
        eval_.kill_imag(ctxt_sign,ctxt_sign)

    return ctxt_sign

def findMaxY(cdict,d,n,t,logN,ndata,context,kpack,sk):
    print("findMaxY...")
    N = math.pow(2,logN)
    ndata_inv = 1/ndata
    m0=heaan.Message(logN-1,0)
    c0=heaan.Ciphertext(context)
    eval_ = heaan.HomEvaluator(context,kpack)
    enc = heaan.Encryptor(context)
    enc.encrypt(m0,kpack,c0)
    dec = heaan.Decryptor(context)
    pow_2_ndata = _get_smallest_pow_2(ndata)
    for i in range(pow_2_ndata):
        m0[i] = 1

    ## BinCount for ys (1,..,t)
    for i in range(1,t+1):
        label = 'label_'+str(i)
        eval_.mult(cdict[label],m0,cdict[label])
        check_boot(cdict[label],eval_)
        print("cdict[label]:")
        print_ctxt(cdict[label],dec,sk,logN-1,pow_2_ndata)
        tmp = Bc(cdict[label],context,kpack,logN,d,n)
        if (i>1):
            eval_.right_rotate(tmp,(i-1),tmp)
        eval_.add(c0,tmp,c0)
    eval_.mult(c0,ndata_inv,c0) 
    print("c0:")
    print_ctxt(c0,dec,sk,logN-1,t)

    ## The returning poisition will be in [0,t-1]
    return findMaxPos(c0,context,kpack,logN,d,n,t,dec,sk)

def print_ctxt_l(dec,sk,logN_1,size, *args):
    m=[]
    for c in args:
        mm=heaan.Message(logN_1)
        dec.decrypt(c,sk,mm)
        m.append(mm)

    for i in range(size):
        l = (mm[i] for mm in m)
        print(i,end=' ')
        for j in l:
            print(j,end=' ')
            if (math.isnan(j.real)):
                print ("nan detected.. stop")
                exit(0)
        print("")

def print_ctxt(c,dec,sk,logN,size):
    m=heaan.Message(logN)
    dec.decrypt(c,sk,m)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
            
def _get_smallest_pow_2(x: int) -> int:
        return 1 << (x - 1).bit_length()

## d: number of X variables
## n: total number of values that a variable can have
## t: total number of values that Y variable can have 
def findMIN2Gini(cdict,d,n,t,logN,context,kpack,sk,ndata):
    eval_ = heaan.HomEvaluator(context,kpack)
    enc = heaan.Encryptor(context)
    dec = heaan.Decryptor(context)
    N = math.pow(2,logN)
    m0 = heaan.Message(logN-1,0)
    ctmpx = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,ctmpx)
    ctmp = heaan.Ciphertext(context)
    pow_2 = _get_smallest_pow_2(t)

    for i in range(1,t+1):
        label = 'label_'+str(i)
        # label_ctxt = heaan.Ciphertext(cdict[label])
        # X_ctxt = heaan.Ciphertext(cdict['X'])
        eval_.mult(cdict[label],cdict['X'],ctmp)
        check_boot(ctmp, eval_)
        
        # left = heaan.Message(logN-1)
        # right = heaan.Message(logN-1)
        left_ctxt = heaan.Ciphertext(context)
        right_ctxt = heaan.Ciphertext(context)
        pow_2_ndata = _get_smallest_pow_2(ndata)
        print(pow_2_ndata)
        for j in range(n*d):
            left = heaan.Message(logN-1,0)
            right = heaan.Message(logN-1,1)
            for k in range(pow_2_ndata*(j+1)):
                left[k]=1
            eval_.sub(right,left,right)
            # ml = [1]*(pow_2_ndata*(j+1))+[0]*(int(N//2)-(pow_2_ndata*(j+1)))
            # mr = [0]*(pow_2_ndata*(j+1))+[1]*(int(N//2)-(pow_2_ndata*(j+1)))
            # left.set_data(data=ml)
            # right.set_data(data=mr)
            
            eval_.mult(ctmp,left,left_ctxt)
            check_boot(left_ctxt, eval_)
            eval_.mult(ctmp,right,right_ctxt)
            check_boot(right_ctxt, eval_)
            cl = Bc(left_ctxt,context,kpack,logN,d,n)
            cr = Bc(right_ctxt,context,kpack,logN,d,n)
            
            sindex = pow_2*j+(i-1)
            eval_.right_rotate(cl,sindex,cl)
            eval_.right_rotate(cr,sindex,cr)
            
            eval_.add(cl,ctmpx,ctmpx)
            eval_.right_rotate(cr,int(N//4),cr)
            eval_.add(cr,ctmpx,ctmpx)

    ## Scaling
    eval_.mult(ctmpx,0.001,ctmpx)
    check_boot(ctmpx, eval_)
    print('scaling ctmpx: ')#22
    print_ctxt(ctmpx,dec,sk,logN,n*d)#22

    sqrt_sum = heaan.Ciphertext(context)
    sum_sqrt = heaan.Ciphertext(context)
    tmp = heaan.Ciphertext(context)
    
    for i in range(0, int(N//2), pow_2):
        m0[i] = 1
    ctxt = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,ctxt)
    
    ## x1+x2+x3
    eval_.square(ctmpx,sqrt_sum)
    i = 1
    while (i<pow_2):
        eval_.left_rotate(sqrt_sum,i,tmp)
        eval_.add(sqrt_sum,tmp,sqrt_sum)
        i*=2
    eval_.mult(sqrt_sum, ctxt, sqrt_sum)
    check_boot(sqrt_sum, eval_)
    print('sqrt_sum: ')#22
    print_ctxt(sqrt_sum,dec,sk,logN,n*d)#22
    
    ## (x1+x2+x3)^2
    i = 1
    while (i<pow_2):
        eval_.left_rotate(ctmpx,i,tmp)
        eval_.add(ctmpx,tmp,ctmpx)
        i*=2
    eval_.mult(ctmpx, ctxt, ctmpx)
    check_boot(sum_sqrt, eval_)
    eval_.square(ctmpx,sum_sqrt)
    print('sum_sqrt: ')#22
    print_ctxt(sum_sqrt,dec,sk,logN,n*d)#22
        
    rl = heaan.Ciphertext(context)
    rr = heaan.Ciphertext(context)
    label_sum = heaan.Ciphertext(context)
    mn=heaan.Message(logN-1,0)
    for i in range(pow_2_ndata):
        mn[i] = 1
    for k in range(1,t+1):
        label = 'label_'+str(k)
        eval_.mult(cdict[label],mn,cdict[label])
        eval_.add(cdict[label], label_sum, label_sum)
    
    label_sum = Bc(label_sum,context,kpack,logN,d,n)
    check_boot(label_sum,eval_)
    print('label sum: ')
    print_ctxt(label_sum,dec,sk,logN,n*d)
    eval_.square(label_sum, label_sum) 
    check_boot(label_sum,eval_)
    label_inverse = heaan.Ciphertext(context)
    heaan.math.approx.inverse(eval_, sum_sqrt, label_inverse)
    check_boot(label_inverse,eval_)       
    
    eval_.sub(sum_sqrt,sqrt_sum,rl)
    print('sum_sqrt - sqrt_sum: ')#22
    print_ctxt(rl,dec,sk,logN,n*d)#22
    eval_.mult(rl, label_inverse, rl)
    print('rl.level: ', rl.level)
    check_boot(rl,eval_)
    
    eval_.right_rotate(rl,int(N//4),rr)
    eval_.add(rl,rr,rl)
    print('left + right: ')#22
    print_ctxt(rl,dec,sk,logN,n*d)#22
    ## rl is currently r=rl+rr
    ##print("r:")
    
    # 필요없는 값 제거
    for i in range(0, n*d*pow_2, pow_2):
        m0[i] = 1
    ctxt = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,ctxt)
    eval_.mult(rl, ctxt, rl)
    check_boot(rl,eval_)
    print('rl: ')#22
    print_ctxt(rl,dec,sk,logN,2*n*d*pow_2)#22

    # 지니 계수 앞에서부터 슬롯 채우기
    tmp = heaan.Ciphertext(context)
    gini = heaan.Ciphertext(context)
    num_iter=0
    for i in range(0, n*d*pow_2, pow_2):
        _m010=[0]*(2**(logN-1))
        _m010[i]=1.0
        m010 = heaan.Message(logN-1)
        for k in range(2**(logN-1)):
            m010[k] = _m010[k]
        # #m010.set_data(_m010)
        eval_.mult(rl, m010, tmp)
        eval_.left_rotate(tmp, (pow_2-1)*num_iter, tmp)
        eval_.add(gini, tmp, gini)
        num_iter+=1
    check_boot(gini,eval_)
   
    print('gini.level: ', gini.level)
    print('지니 계수: ')
    print_ctxt(gini,dec,sk,logN,d*n*pow_2)

    cmin = findMinPos(gini,context,kpack,logN,d,n,d*n,dec,sk)
    check_boot(cmin,eval_) 
    mdn = heaan.Message(logN-1,0.0)
    for i in range(n*d): 
        mdn[i]=1.0
    eval_.mult(cmin,mdn,cmin)
    check_boot(cmin,eval_) 
    print("cmin end:")
    print_ctxt(cmin,dec,sk,logN,n*d*pow_2)
    
    ## extend the 1 by n-1
    bl = int(math.log2(n)) 
    bm = {}
    tn = n
    ct=0
    while (tn>1):
        if (tn % 2 == 1):
            bm[ct]=1
        else:
            bm[ct]=0
        ct+=1
        tn=tn//2

    bm[ct]=tn

    m0 = heaan.Message(logN-1,0)
    c_ext = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,c_ext)
    print("c_ext:")
    print("init....")
    i=ct
    sdind=0
    while (i>=0):
        if bm[i]==1:
            ind=0
            s=1
            ctmp1 = heaan.Ciphertext(cmin)
            print('(s)i=',i)
            print_ctxt(ctmp1,dec,sk,logN,n*d+8)
            while(ind<i):
                eval_.left_rotate(ctmp1,s,ctmp)
                eval_.add(ctmp1,ctmp,ctmp1)
                s*=2
                ind+=1
            if (sdind>0):
                eval_.left_rotate(ctmp1,sdind,ctmp1)
            eval_.add(c_ext,ctmp1,c_ext)
            sdind+=(s)
            print('i=',i)
            print_ctxt(c_ext,dec,sk,logN,n*d+8)
        i-=1

    ## making c_a, c_v
    m1 = heaan.Message(logN-1,0)
    for i in range(n):
        m1[i]=1.0
    m01 = heaan.Message(logN-1,0)
    m01[0]=1.0
    
    print("============================= making c_a c_v ====================")
    print('c_ext: ')
    print_ctxt(c_ext,dec,sk,logN,n*d+8)
    c_a = heaan.Ciphertext(context)
    c_v = heaan.Ciphertext(context)
    tmp = heaan.Ciphertext(context)
    for i in range(d):
        if (i>0):
            eval_.left_rotate(c_ext,n*i,ctmp)
        else:
            ctmp = heaan.Ciphertext(c_ext)
            
        #making c_v
        print('ctmp for c_v: ')
        print_ctxt(ctmp,dec,sk,logN,n*d)
        eval_.mult(ctmp,m01,tmp)
        print('tmp: ')
        print_ctxt(tmp,dec,sk,logN,n*d)
        check_boot(tmp,eval_)
        # tmp_rt = heaan.Ciphertext(context)
        # 여기 로테이트 하는거 이진수 이용해서 하면 rotate 횟수 줄일수 있어
        # r=1
        # while(r<int(N//2)):
        #     eval_.right_rotate(tmp,r,tmp_rt)
        #     eval_.add(tmp,tmp_rt,tmp)
        #     r*=2
        ctmp2 = heaan.Ciphertext(context)
        m0 = heaan.Message(logN-1,0)
        bin_ext = heaan.Ciphertext(context)
        enc.encrypt(m0,kpack,bin_ext)
        print("bin_ext:")
        print("init....")
        r=ct
        sdind=0
        print(r, ct, bm)
        while (r>=0):
            if bm[r]==1:
                ind=0
                s=1
                ctmp1 = heaan.Ciphertext(tmp)
                print('(s)r=',r)
                print_ctxt(ctmp1,dec,sk,logN,n*d+8)
                while(ind<r):
                    eval_.right_rotate(ctmp1,s,ctmp2)
                    eval_.add(ctmp1,ctmp2,ctmp1)
                    s*=2
                    ind+=1
                if (sdind>0):
                    eval_.right_rotate(ctmp1,sdind,ctmp1)
                eval_.add(bin_ext,ctmp1,bin_ext)
                sdind+=(s)
                print('r=',r)
                print_ctxt(bin_ext,dec,sk,logN,n*d+8)
            r-=1
        
        print('bin_ext: ')
        print_ctxt(bin_ext,dec,sk,logN,n*d)
        eval_.mult(ctmp,bin_ext,bin_ext)
        print('ctmp end: ')
        print_ctxt(ctmp,dec,sk,logN,n*d)
        check_boot(bin_ext,eval_)
        eval_.add(c_v,bin_ext,c_v)  
        #eval_.mult(c_v,m1,c_v)   
        check_boot(c_v,eval_)
        #making c_a
        print('ctmp for c_a: ')
        print_ctxt(ctmp,dec,sk,logN,n*d)
        eval_.mult(ctmp,m01,ctmp)
        check_boot(ctmp,eval_)
        eval_.right_rotate(ctmp, i, ctmp)
        eval_.add(c_a, ctmp, c_a)
        
               
        
        
    # eval_.mult(c_a,csum,c_a)
    # check_boot(c_a, eval_)

    # eval_.mult(c_v,csum,c_v)
    # check_boot(c_v, eval_)
    
    print("ca")
    print_ctxt(c_a,dec,sk,logN,d) 

    print("cv")
    print_ctxt(c_v,dec,sk,logN,n)
    print('====================== ca cv end =========================')
    return c_a,c_v

def DT_Learn(L,cdict,d,n,t,logN,context,kpack,sk,ndata,model_path):
    dec = heaan.Decryptor(context)
    eval_ = heaan.HomEvaluator(context,kpack)
    enc = heaan.Encryptor(context)

    print('=========== findMIN2Gini ==========', L.id)
    L.ca,L.cv = findMIN2Gini(cdict,d,n,t,logN,context,kpack,sk,ndata)
    print('=========== findMIN2Gini END ==========', L.id)
    print('=========== findMaxY ==========', L.id)
    L.cy = findMaxY(cdict,d,n,t,logN,ndata,context,kpack,sk)
    print('=========== findMaxY END ==========', L.id)
    L.ca.save(model_path+L.id+"_ca.ctxt")
    L.cv.save(model_path+L.id+"_cv.ctxt")
    ## Making b_y, need make sure that c_y has only one 1 or all zeros
    ## Otherwise the below routine is not working correctly
    m1 = heaan.Message(logN-1,0)
    m1[0]=1.0
    m0=heaan.Message(logN-1,0)
    enc = heaan.Encryptor(context)
    cacc = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,cacc)
    ctmp = heaan.Ciphertext(context)
    ctmp2 = heaan.Ciphertext(context)
    for l in range(t):
        if (l==0):
            eval_.mult(L.cy,m1,ctmp)
        else:
            eval_.left_rotate(L.cy,l,ctmp)
            eval_.mult(ctmp,m1,ctmp)
        i=1
        while i<pow(2,logN-2):
            eval_.right_rotate(ctmp,i,ctmp2)
            eval_.add(ctmp,ctmp2,ctmp)
            i*=2
        eval_.add(cacc,ctmp,cacc)

    L.by = heaan.Ciphertext(cacc)

    value = heaan.Ciphertext(context) # y값의 개수
    sample = heaan.Ciphertext(context)
    sample_inverse = heaan.Ciphertext(context) # y값 개수의 총합
    enc.encrypt(m0,kpack,value)
    enc.encrypt(m0,kpack,sample)
    enc.encrypt(m0,kpack,sample_inverse)
    val = heaan.Message(logN-1,0)
    pow_2_ndata = _get_smallest_pow_2(ndata)
    for i in range(pow_2_ndata):
        val[i] = 1
    for i in range(1,t+1):
        label = 'label_'+str(i)
        eval_.mult(cdict[label],val,cdict[label])
        c_val = Bc(cdict[label],context,kpack,logN,d,n)
        if (i>1):
            eval_.right_rotate(c_val,(i-1),c_val)
        eval_.add(value,c_val,value)
    print('Value: ')
    print_ctxt(value,dec,sk,logN-1,t)

    tmp = heaan.Ciphertext(context)
    r = 1
    while r < pow(2, logN-1):
        eval_.left_rotate(value, r, tmp)
        eval_.add(value, tmp, value)
        r *= 2
    eval_.mult(value, m1, sample)
    check_boot(sample,eval_)
    print('Sample: ')
    print_ctxt(sample,dec,sk,logN-1,1)
    
    heaan.math.approx.inverse(eval_, sample, sample_inverse)
    check_boot(sample_inverse, eval_)
    #22print('Sample inverse: ')
    #22print_ctxt(sample_inverse,dec,sk,logN-1,1)
    eval_.mult(sample, sample_inverse, sample_inverse)
    check_boot(sample_inverse, eval_)
    eval_.mult(L.by, sample_inverse, L.by)
    check_boot(L.by, eval_)
    
    by_rt = heaan.Ciphertext(context)
    i=1
    while (i<2**(logN-1)):
        eval_.right_rotate(L.by, i, by_rt)
        eval_.add(L.by, by_rt, L.by)
        i*=2
    check_boot(L.by, eval_)
        
    save_cyby = heaan.Ciphertext(context)
    save_cy = heaan.Ciphertext(context)
    save_by = heaan.Ciphertext(context)
    eval_.right_rotate(L.cy, 1, save_cy)
    eval_.mult(L.by, m1, save_by)
    check_boot(save_by, eval_)
    eval_.add(save_by, save_cy, save_cyby)
    save_cyby.save(model_path+L.id+"_cyby.ctxt")
    print('ca: ', L.id)
    print_ctxt(L.ca,dec,sk,logN-1,d)
    print('cv: ', L.id)
    print_ctxt(L.cv,dec,sk,logN-1,n)
    print('cy: ', L.id)
    print_ctxt(L.cy,dec,sk,logN-1,t)
    print('by: ', L.id)
    print_ctxt(L.by,dec,sk,logN-1,t)
    exit()

def Make_data(L,cdict,d,n,t,logN,context,kpack,sk,ndata,model_path,ctxt_path):
    dec = heaan.Decryptor(context)
    eval_ = heaan.HomEvaluator(context,kpack)
    enc = heaan.Encryptor(context)
    m1 = heaan.Message(logN-1,0)
    m1[0]=1.0
    m0=heaan.Message(logN-1,0)
    ctmp = heaan.Ciphertext(context)
    ca = heaan.Ciphertext(context)
    cv = heaan.Ciphertext(context)
    parent_id = L.id[:len(L.id)-1]
    ca.load(model_path+parent_id+'_ca.ctxt')
    cv.load(model_path+parent_id+'_cv.ctxt')

    Xd = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,Xd)
    rotate_ca = heaan.Ciphertext(context)
    rotate_cv = heaan.Ciphertext(context)
    pow_2_ndata = _get_smallest_pow_2(ndata)
    rot = heaan.Ciphertext(context)
    child = heaan.Ciphertext(context)
    enc.encrypt(m0,kpack,child)
    _cv = heaan.Ciphertext(context)
    mn = heaan.Message(logN-1)
    for i in range(n):
        mn[i] = 1
    enc.encrypt(mn,kpack,_cv)
    eval_.sub(_cv,cv,_cv)
    idx = 'X'
    print("cdict[X]",L.id)
    print_ctxt(cdict[idx],dec,sk,16,32768)
    
    for i in range(d):
        eval_.left_rotate(ca,i,rotate_ca)
        eval_.mult(rotate_ca,m1,rotate_ca)
        check_boot(rotate_ca, eval_)
        r=1
        while (r<2**(logN-1)):
            eval_.right_rotate(rotate_ca,r,rot)
            eval_.add(rotate_ca,rot,rotate_ca)
            r*=2

        eval_.left_rotate(cdict[idx],pow_2_ndata*n*i,ctmp)
        eval_.mult(ctmp,rotate_ca,ctmp)
        eval_.add(Xd,ctmp,Xd)
        print("Xd",L.id,i+1)
        print_ctxt(Xd,dec,sk,16,32768)
    print("Xd END",L.id)
    print_ctxt(Xd,dec,sk,16,32768)
    
    if L.id[-1] == 'l':
        ## left branch   
        for j in range(n):
            eval_.left_rotate(cv,j,rotate_cv)
            eval_.mult(rotate_cv,m1,rotate_cv)
            check_boot(rotate_cv, eval_)
            r=1
            while (r<pow_2_ndata):
                eval_.right_rotate(rotate_cv,r,rot)
                eval_.add(rotate_cv,rot,rotate_cv)
                r*=2
            print("rotate cv",j)
            print_ctxt(rotate_cv,dec,sk,16,pow_2_ndata)
            eval_.left_rotate(Xd,pow_2_ndata*j,ctmp)
            eval_.mult(ctmp,rotate_cv,ctmp)
            check_boot(ctmp, eval_)
            eval_.add(child,ctmp,child)
        print("child",L.id,child.level)
        print_ctxt(child,dec,sk,16,pow_2_ndata)
            
        r=1
        while (pow_2_ndata*r<2**(logN-1)):
            eval_.right_rotate(child,pow_2_ndata*r,rot)
            eval_.add(child,rot,child)
            r*=2
        print("sel_left",L.id,child.level)
        print_ctxt(child,dec,sk,16,32768)
        
        eval_.mult(cdict[idx],child,ctmp)
        check_boot(ctmp, eval_)
        ctmp.save(ctxt_path+L.id+'/'+idx+'.ctxt')
        
        for i in range(1,t+1):
            ind='label_'+str(i)
            eval_.mult(cdict[ind],child,ctmp)
            check_boot(ctmp, eval_)
            ctmp.save(ctxt_path+L.id+'/'+ind+'.ctxt')
            
    else:         
        ## right branch
        for j in range(n):
            eval_.left_rotate(_cv,j,rotate_cv)
            eval_.mult(rotate_cv,m1,rotate_cv)
            check_boot(rotate_cv, eval_)
            r=1
            while (r<pow_2_ndata):
                eval_.right_rotate(rotate_cv,r,rot)
                eval_.add(rotate_cv,rot,rotate_cv)
                r*=2
            print("rotate cv",j)
            print_ctxt(rotate_cv,dec,sk,16,pow_2_ndata)
            eval_.left_rotate(Xd,j,ctmp)
            eval_.mult(ctmp,rotate_cv,ctmp)
            check_boot(ctmp, eval_)
            eval_.add(child,ctmp,child)
        print("child",L.id,child.level)
        print_ctxt(child,dec,sk,16,pow_2_ndata)
            
        r=1
        while (pow_2_ndata*r<2**(logN-1)):
            eval_.right_rotate(child,pow_2_ndata*r,rot)
            eval_.add(child,rot,child)
            r*=2
        print("sel_right",L.id,child.level)
        print_ctxt(child,dec,sk,16,32768)
        
        eval_.mult(cdict[idx],child,ctmp)
        check_boot(ctmp, eval_)
        ctmp.save(ctxt_path+L.id+'/'+idx+'.ctxt')
        
        for i in range(1,t+1):
            ind='label_'+str(i)
            eval_.mult(cdict[ind],child,ctmp)
            check_boot(ctmp, eval_)
            ctmp.save(ctxt_path+L.id+'/'+ind+'.ctxt')
    
    for i in range(1,t+1):
        ind='label_'+str(i)   
        print("label",L.id)
        print_ctxt(ctmp,dec,sk,16,32768) 
            
    return    
           
