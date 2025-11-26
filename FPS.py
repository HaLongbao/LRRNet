import cv2
import torch
import torch.nn.functional as F
import scipy.io as scio
import os
import time

from models import get_model

dataset_name = 'IRSTD-1k'  # NUDT-SIRST IRSTD-1k sirst_aug SIRSTv1
net = get_model('LRRNet')

file_path = r"/home/avp/avpnet/RPCANet/datasets/IRSTD-1k/test/images/"

if dataset_name == 'IRSTD-1k':
    pkl_file = '/home/avp/avpnet/LRRNet/checkpoints/IRDST_1K_miou_0.7239356450020329.pkl'

checkpoint = torch.load(pkl_file, map_location=torch.device('cuda:0'))
net.load_state_dict(checkpoint)
net.eval()
net.to('cuda:0')

imgDir = f"./result/{dataset_name}/img1/"
os.makedirs(imgDir, exist_ok=True)
matDir = f"./result/{dataset_name}/mat1/"
os.makedirs(matDir, exist_ok=True)

number = 0
total_time = 0.0  # 用于累计推理时间

if dataset_name == 'SIRSTv1':
    txt_name = "test_v1.txt"
    txt_path = fr'E:\datasets\SIRSTdevkit-master\Splits\{txt_name}'
    file_path_base = fr'E:\datasets\SIRSTdevkit-master\PNGImages'
    with open(os.path.join(txt_path), "r") as f:
        file_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
else:
    file_list = os.listdir(file_path)

for filename in file_list:
    number += 1
    if dataset_name == 'SIRSTv1':
        filename = filename + '.png'
        img_gray = cv2.imread(file_path_base + '/' + filename, 0)
    else:
        img_gray = cv2.imread(file_path + '/' + filename, 0)

    img_gray = cv2.resize(img_gray, [512, 512], interpolation=cv2.INTER_LINEAR)
    img = img_gray.reshape(1, 1, 512, 512) / 255.0
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.to('cuda:0')

    name = os.path.splitext(filename)[0]
    matname = name + '.mat'

    with torch.no_grad():
        start = time.time()
        D, T = net(img)
        end = time.time()
        elapsed = end - start
        total_time += elapsed

        T = torch.sigmoid(T)  # F.sigmoid 已弃用
        T = T.detach().cpu().numpy().squeeze()
        T[T < 0] = 0
        Tout = (T * 255).astype('uint8')

    cv2.imwrite(os.path.join(imgDir, filename), Tout)
    scio.savemat(os.path.join(matDir, matname), {'T': T})
    print(f"[{number}/{len(file_list)}] {filename} processed, time: {elapsed:.4f}s")

# 计算平均 FPS
# 请注意，FPS存在一定的波动，特别是第一帧处理，模型初始化等速度较慢，可以选择跳过第一帧的计时
avg_fps = number / total_time
print(f"Processed {number} images, Average FPS: {avg_fps:.2f}")
