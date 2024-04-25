import DTmodule_final as mod
import pandas as pd
import piheaan as heaan
import random
import os
import json
import sys

df = pd.read_csv('iris_sturges_p.csv')
logN=16
params = heaan.ParameterPreset.FGb
context = heaan.make_context(params)
heaan.make_bootstrappable(context)

# #key 생성
# sk = heaan.SecretKey(context)
# key_file_path = "./newkey"
# os.makedirs(key_file_path, mode=0o775, exist_ok=True)
# log_num_slot = heaan.get_log_full_slots(context)
# num_slot = 1 << log_num_slot
# sk.save(key_file_path+"/secretkey.bin")

# sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin")
# key_generator = heaan.KeyGenerator(context, sk)
# key_generator.gen_common_keys()
# key_generator.save(key_file_path + "/")

# key 이미 생성
sk = heaan.SecretKey(context)
key_file_path = "./newkey"
log_num_slot = heaan.get_log_full_slots(context)

num_slot = 1 << log_num_slot
sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin")
keypack = heaan.KeyPack(context, key_file_path+"/")
keypack.load_enc_key()
keypack.load_mult_key()
eval = heaan.HomEvaluator(context,keypack)
dec = heaan.Decryptor(context)
enc = heaan.Encryptor(context)

# 암호문 처리
ctxt_path = './enc_data4/'
model_path = './model_ctxt2/'
depth = sys.argv[2]
# os.system('rm -f ./enc_data3/*.ctxt') #노드 데이터 정보 초기화 하는 부분
# os.mkdir(ctxt_path+'o/')
# mod.encrypt_and_save(df, ctxt_path+'o/',context, keypack)
# out_cdict = mod.load_ctxt(df.columns[1:],ctxt_path+'o/',context)
# file_list = mod.make_file_list(ctxt_path+'o/')
# out_cdict = mod.load_ctxt(file_list,ctxt_path+'o/',context)
# print(out_cdict)

#JSON 파일 저장하기 => META Data
json_path = "./JSON_TEST/"
mod.save_metadata_json('iris_sturges.csv', depth, json_path)
# exit()
# json 파일 불러오기. 이 과정을 안거치면 json 파일에 저장된 내용을 불러올 수 없다.
with open(json_path + 'Metadata.json') as f:
    meta = json.load(f)

# 확인하기
# for i in ['ndata', 'n', 'd', 't', 'depth', 'node']:
#     print(f"{i} : ", meta[i])


L = mod.Node(meta['d'],meta['n'])
L.id = sys.argv[1]

if len(L.id) != 1:
    os.mkdir(ctxt_path+L.id)
    parent_id = L.id[:len(L.id)-1]
    # 부모 노드 데이터를 불러오기
    file_list = mod.make_file_list(ctxt_path+parent_id+'/')
    cdict = mod.load_ctxt(file_list,ctxt_path+parent_id+'/',context)
    mod.Make_data(L,cdict,meta['d'],meta['n'],meta['t'],logN,context,keypack,sk,meta['ndata'],model_path,ctxt_path)
else:
    os.mkdir(ctxt_path+L.id)
    mod.enc_and_save(df, meta['n'], meta['d'], meta['ndata'], logN, ctxt_path+L.id+'/', context, keypack)
        
        
file_list = mod.make_file_list(ctxt_path+L.id+'/')
out_cdict = mod.load_ctxt(file_list,ctxt_path+L.id+'/',context)
mod.DT_Learn(L,out_cdict,meta['d'],meta['n'],meta['t'],logN,context,keypack,sk,meta['ndata'],model_path)
