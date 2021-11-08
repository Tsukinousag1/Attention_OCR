import cv2
import os
from skimage.transform import resize as imresize
from crnn import Attention_ocr
import time
import glob
import torch
import numpy as np
import json,argparse
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--img',type=str,default='/home/std2021/hejiabang/OCR/Attention/check_pictures',help='the image we want to predict')
parser.add_argument('--input-h',type=int,default=32,help='the height of the input image to network')
parser.add_argument('--input_w',type=int,default=100,help='the width of the input image to network')
parser.add_argument('--use_gpu',action='store_true',default=True,help='enable cuda')
parser.add_argument('--index_to_char',type=str,default='/home/std2021/hejiabang/OCR/Attention/index_to_char.json',help='index_to_char')
parser.add_argument('--checkpoints',type=str,default='/home/std2021/hejiabang/OCR/Attention/checkpoints/2_checkpoint.pt',help='checkpoints model directory')
opt=parser.parse_args()

if __name__ == '__main__':
    with open(opt.index_to_char,'r',encoding='utf-8') as f:
        index_to_char=json.load(f)

    n_class=len(index_to_char)
    net=Attention_ocr(use_gpu=opt.use_gpu,NUM_CLASS=n_class)
    net.load_state_dict(torch.load(opt.checkpoints)['state_dict'])#此处不要偷懒后面的一句话
    if torch.cuda.is_available():
        net=net.cuda()
    net=net.eval()
    print('==== 开始识别 =====')
    img_path=os.listdir(opt.img)
    #可以看到glob.glob(file_glob)的结果就是符合相应模式的文件列表，
    #该函数对大小写不敏感，.jpg与.JPG是一样的
    for path in img_path:
        new_img_path=opt.img+'/'+path
        img=cv2.imread(new_img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.imshow(img,cmap='gray')
        img=img.astype('float32')/127.5-1
        if img.ndim==2:
            img=np.stack([img]*3,-1)
            #h,w ——> h,w,c
        """a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        a, a.shape
        (tensor([[1., 2., 3.],
                 [4., 5., 6.]]),
         torch.Size([2, 3]))
        b = torch.stack([a] * 3, -1)
        b, b.shape
        (tensor([[[1., 1., 1.],
                  [2., 2., 2.],
                  [3., 3., 3.]],

                 [[4., 4., 4.],
                  [5., 5., 5.],
                  [6., 6., 6.]]]),
         torch.Size([2, 3, 3]))"""
        img=imresize(img,(opt.input_h,opt.input_w),mode='constant')
        img=torch.from_numpy(img.transpose([2,0,1]).astype(np.float32))[None,...]
        #x = torch.Tensor([1, 2, 3, 4, 5, 6])
        #x[None, ...].shape
        #torch.Size([1, 6])
        if torch.cuda.is_available():
           img=img.cuda()

        t=time.time()
        output=net(img)
        #torch.Size([1, 5560, 10])
        output=output.max(1)[1].squeeze(0)
        #[1,10]——>[10]

        text=''.join([index_to_char[str(_)] for _ in output.tolist()])
        print('Path: ',path,'\t====>>>>\t',text,'time cost: %3f' % (time.time()-t))
        plt.show()




















