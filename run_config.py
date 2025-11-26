import os
os.environ['CUDA_VISIBLE_DEVICE'] ='3'
gpu = '--gpu 3' 

train_cfg = ' --epoch 400 --lr 1e-4 --save-iter-step 100 --log-per-iter 100 '
data_irstd1k = ' --dataset irstd1k '
data_sirstaug = ' --dataset sirstaug '
data_nudt = ' --dataset nudt '

for i in range(1):
    os.system('python train.py --net-name LRRNet --batch-size 8 --base-dir train_logs0' + train_cfg + data_irstd1k + gpu)


