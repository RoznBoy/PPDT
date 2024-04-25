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
import treeModule as mod

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

# 암호문 처리
ctxt_path = './enc_data/'
model_path = './model_ctxt/'
save_path = sys.argv[4]
json_path = "./JSON/"
depth = int(sys.argv[2])

#JSON 파일 저장하기 => META Data
with open(json_path + 'Metadata.json') as f:
    meta = json.load(f)
    
# node dictionary 만들기
name_dict = {}
def node_list(name, depth):
    if len(name) in name_dict.keys():
        name_dict[len(name)].append(name)
    else:    
        name_dict[len(name)] = [name]
        
    if len(name) == depth+1:
        return
    
    new = name+'l'
    node_list(new, depth)
    new = name+'r'
    node_list(new, depth)
    
    return name_dict

root = 'o'
node_list(root, depth)

start = time.time()
L = mod.Node(meta['n'],meta['d'])
print('노드 정보 time: %.5f sec' %(time.time()-start))
num_gini = meta['d']*meta['n']
node_key = int(sys.argv[1])
_node = name_dict[node_key]
#print('node list', _node)

## parameter for minPos maxPos
n_slot = mod.n_slot_func(num_gini)
nslot = mod.n_slot_func(meta['t'])
pow_2_ndata = mod._get_smallest_pow_2(meta['ndata'])
#print(n_slot, nslot, n_slot, nslot)

## parameter for precomputation 
interval = meta['n']*meta['d']*(2**(depth-(node_key-1)))
leaf_ctxt_num = mod._get_smallest_pow_2(int((np.ceil((meta['n']*meta['d']*(2**depth))/num_slot)))) # leaf node level 암호문 수
one_ctxt_leaf_node = int(2**(depth)/leaf_ctxt_num) # leaf node 하나의 암호문에 들어가는 노드 수
one_ctxt_node = int(np.ceil(one_ctxt_leaf_node/(2**(depth-(node_key-1))))) # level 별 하나의 암호문에 들어가는 노드 수 
parent_one_ctxt_node = int(np.ceil(one_ctxt_leaf_node/(2**(depth-(node_key-2))))) # 부모 level 별 하나의 암호문에 들어가는 노드 수 
parent_ctxt_num = int((2**(node_key-2))//parent_one_ctxt_node) # 부모 level 암호문 수

m0 = heaan.Block(context,encrypted = False, data = [0]*num_slot)

m = [1]*(meta['t']) + [0]*(num_slot-num_gini)
m_cy = heaan.Block(context,encrypted = False, data = m)

cy_list = []
by_list = []
num_list = []

if n_slot*len(_node) <= num_slot:
    y_total = m0.encrypt()
    for i in range(len(_node)):
        L.id = _node[i]
        
        #print('L.id :', L.id)
        if len(L.id) != 1:
            os.mkdir(ctxt_path+L.id)
            parent_id = L.id[:len(L.id)-1]
            #print('parent_id:', parent_id)
            # 부모 노드 데이터를 불러오기
            file_list = mod.make_file_list(ctxt_path+parent_id+'/')
            start = time.time()
            cdict = mod.load_ctxt(file_list,ctxt_path+parent_id+'/')
            print(parent_id, 'make parent_cdict time: %.5f sec' %(time.time()-start))
            start = time.time()
            mod.update_data(L,parent_id,cdict,meta['d'],meta['n'],meta['t'],meta['ndata'],model_path,ctxt_path)
            print(L.id, 'Update data time: %.5f sec' %(time.time()-start))

        file_list = mod.make_file_list(ctxt_path+L.id+'/')
        start = time.time()
        out_cdict = mod.load_ctxt(file_list,ctxt_path+L.id+'/') 
        print(L.id, 'make cdict time: %.5f sec' %(time.time()-start))

        start = time.time()
        value = mod.findMaxY(out_cdict,meta['d'],meta['n'],meta['t'],meta['ndata'])
        print(L.id, 'findMaxY time: %.5f sec' %(time.time()-start))
        value = value.__rshift__(nslot*i)
        y_total = value + y_total

    start = time.time()
    cy_sum = mod.findMaxPosMany(y_total,meta['d'],meta['n'],meta['t'],nslot,len(_node))
    print(len(_node), 'findMaxPosMany time: %.5f sec' %(time.time()-start))
 
    save_cy = m0.encrypt()
    save_by = m0.encrypt()
    cnt = 0
    for i in range(len(_node)):
        L.id = _node[i]

        cy = cy_sum.__lshift__(nslot*i)
        cy = m_cy * cy
        # print('DT_Learn 시작 전 cy: ')
        # mod.print_ctxt(cy,dec,sk,logN,nslot)
        
        file_list = mod.make_file_list(ctxt_path+L.id+'/')
        out_cdict = mod.load_ctxt(file_list,ctxt_path+L.id+'/')
        
        start = time.time() 
        Lcy,Lby = mod.DT_Learn_leaf(L,out_cdict,cy,meta['d'],meta['n'],meta['t'],meta['ndata'],model_path)
        print(L.id, 'DT_Learn time: %.5f sec' %(time.time()-start))
        
        start = time.time() 
        Lcy = Lcy.__rshift__(interval*(i%one_ctxt_node))
        save_cy = Lcy + save_cy
        Lby = Lby.__rshift__(interval*(i%one_ctxt_node))
        save_by = Lby + save_by
        print(L.id, 'step7 time: %.5f sec' %(time.time()-start))
        
        cnt += 1
        if cnt == one_ctxt_node:
            cy_list.append(save_cy)
            by_list.append(save_by)
            num_list.append(cnt)
            cnt = 0

            save_cy = m0.encrypt()
            save_by = m0.encrypt()

    start = time.time()
    for i in range(len(num_list)):
        tmpCy = mod.UpdateCy(model_path,node_key,cy_list[i],by_list[i],interval,num_list,i,parent_ctxt_num,meta['d'],meta['n'],meta['t'],depth)
        tmpCy.save(save_path+"cy"+str(i)+".ctxt") ## leaf node
    print(len(_node), 'leaf_node_evaluation 하기 위한 암호문 저장 시간: %.5f sec' %(time.time()-start))  
    
####################################################################################
###################### findMinPosMany 한번에 못하니까 만든 부분 ###########################
####################################################################################    
else:
    save_cy = m0.encrypt()
    save_by = m0.encrypt()

    max_min4 = int(num_slot//n_slot)
    cnt = 0
    for k in range(int(np.ceil(len(_node)/max_min4))):
        y_total = m0.encrypt()
        small_node = _node[max_min4*k:max_min4*(k+1)]
        # print(small_node)
        for i in range(len(small_node)):
            L.id = small_node[i]
            if len(L.id) != 1:
                os.mkdir(ctxt_path+L.id)
                parent_id = L.id[:len(L.id)-1]
                #print('parent_id:', parent_id)
                # 부모 노드 데이터를 불러오기
                file_list = mod.make_file_list(ctxt_path+parent_id+'/')
                start = time.time()
                cdict = mod.load_ctxt(file_list,ctxt_path+parent_id+'/')
                print(parent_id, 'make parent_cdict time: %.5f sec' %(time.time()-start))
                start = time.time()
                mod.update_data(L,parent_id,cdict,meta['d'],meta['n'],meta['t'],meta['ndata'],model_path,ctxt_path)
                print(L.id, 'Update data time: %.5f sec' %(time.time()-start))
                
            file_list = mod.make_file_list(ctxt_path+L.id+'/')
            start = time.time()
            out_cdict = mod.load_ctxt(file_list,ctxt_path+L.id+'/') 
            print(L.id, 'make cdict time: %.5f sec' %(time.time()-start))

            start = time.time()
            value = mod.findMaxY(out_cdict,meta['d'],meta['n'],meta['t'],meta['ndata'])
            print(L.id, 'findMaxY time: %.5f sec' %(time.time()-start))
            value = value.__rshift__(nslot*i)
            y_total = value + y_total

        start = time.time()
        cy_sum = mod.findMaxPosMany(y_total,meta['d'],meta['n'],meta['t'],nslot,len(_node))
        print(len(_node), 'findMaxPosMany time: %.5f sec' %(time.time()-start))
    

        for i in range(len(small_node)):
            L.id = small_node[i]

            cy = cy_sum.__lshift__(nslot*i)
            cy = m_cy * cy
            
            file_list = mod.make_file_list(ctxt_path+L.id+'/')
            out_cdict = mod.load_ctxt(file_list,ctxt_path+L.id+'/')
            
            start = time.time() 
            Lcy,Lby = mod.DT_Learn(L,out_cdict,cy,meta['d'],meta['n'],meta['t'],meta['ndata'],model_path)
            print(L.id, 'DT_Learn time: %.5f sec' %(time.time()-start))
            
            start = time.time()
            Lcy = Lcy.__rshift__(interval*int((max_min4*k+i)%one_ctxt_node))
            save_cy = Lcy + save_cy
            Lby = Lby.__rshift__(interval*int((max_min4*k+i)%one_ctxt_node))
            save_by = Lby + save_by
            print(L.id, 'Make evaluation rotate time: %.5f sec' %(time.time()-start))
            
            cnt += 1
            if cnt == one_ctxt_node:
                cy_list.append(save_cy)
                by_list.append(save_by)
                num_list.append(cnt)
                cnt = 0
                
                save_cy = m0.encrypt()
                save_by = m0.encrypt()

    start = time.time()
    for i in range(len(num_list)):
        tmpCy = mod.UpdateCy(model_path,node_key,cy_list[i],by_list[i],interval,num_list,i,parent_ctxt_num,meta['d'],meta['n'],meta['t'],depth)
        tmpCy.save(save_path+"cy"+str(i)+".ctxt") ## leaf node
    print(len(_node), 'leaf_node_evaluation 하기 위한 암호문 저장 시간: %.5f sec' %(time.time()-start))  

exit()
