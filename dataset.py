import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.transform import resize as imresize

import torch
import cv2
import json


class CNDATA(Dataset):
    def __init__(self,img_base,img_transforms,label_transforms):
        #img_base:img位置与label信息
        super(CNDATA, self).__init__()
        self.img_base=img_base
        self.img_transforms=img_transforms
        self.label_transforms=label_transforms

    def __getitem__(self, index):
        #./images/33069953_4129036931.jpg 到此刻，不要煮的时间
        info=self.img_base[index].split(' ')
        img_path=info[0]
        img=cv2.imread(img_path)
        #print(img_path,type(img),img.shape)
        #H,W,_=img.shape
        if img is not None:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #先转为灰度二值图像
            #将像素值为[0,255]之间的数转化为[-1,1]
            img=img.astype('float')/127.5-1
            if img.ndim==2:
                #2维度数组，即灰度图像，改为三维图像
                img=np.stack([img]*3,-1)#[W,H,3]
            label=info[1]
            if self.img_transforms:
                img=self.img_transforms(img)
            if self.label_transforms:
                label=self.label_transforms(label)
            return img,label
        else:
            print(img_path)

    def __len__(self):
        return len(self.img_base)

#one_hot
def index_to_onehot(idx,num_class):
    # idx->a list of int indices,like[1,3,6,9]
    # num_class->number of classes
    assert (max(idx)<num_class)
    return (np.arange(num_class)==np.array(idx)[:,None])
"""idx=[ 0 , 1 , 2 , 3 ]
num_class=5
b=index_to_onehot(idx,num_class)
print(b)"""
"""
[[ True False False False False]
 [False  True False False False]
 [False False  True False False]
 [False False False  True False]]"""

def label_transforms(char_index,num_class,max_seq_len):
    def str_to_index(label):
        #str映射到数字
        return [char_index[_] for _ in label]
    def pad_label(index_label):
        #padding label [...]+[36]
        diff_w=max_seq_len-len(index_label)
        return np.array(index_label+[num_class-1]*diff_w)
    return transforms.Compose([
        transforms.Lambda(str_to_index),#str映射为数字
        transforms.Lambda(pad_label),#填充36到最大长度,其中末尾位是连续的eof
        transforms.Lambda(lambda label:index_to_onehot(label,num_class)),#对得到的label进行one_hot编码
        transforms.Lambda(lambda label:label.transpose([1,0]).astype(np.float32)),#行列交换
        transforms.Lambda(lambda label:torch.from_numpy(label)),#转换为Tensor
    ])

"""d=np.array([0,1,2,3]+[36]*4)
print(d)
[ 0  1  2  3 36 36 36 36]"""

def cn_transform(input_h,input_w):
    def resize_with_ratio(x):
        #[img , outshape]
        return imresize(x,(input_h,input_w),mode='constant')
    #[input_h,input_w,input_c]

    return transforms.Compose([
        transforms.Lambda(resize_with_ratio),
        transforms.Lambda(lambda x:x.transpose([2,0,1]).astype(np.float32)),
        #[input_c,input_h,input_w]
        transforms.Lambda(lambda x:torch.from_numpy(x))
    ])

def get_label(label_path):
    with open(label_path,'rb') as f:
        lines=f.readlines()
        #print(lines)
        #print(type(lines[0]))
        #[b'20455828_2605100732.jpg 263 82 29 56 35 435 890 293 126 129\r\n'...
        #<class 'bytes'>

    return np.array([_.decode('utf-8').strip() for _ in lines])
    #<class 'str'>
#去除首尾字符
"""label_path="../data/OCR-data/tra.txt"
c=get_label(label_path)
print(c)"""
#['/media/chenjun/ed/23_OCR/Synthetic_Chinese_String_Dataset/Synthetic_Chinese_String_Dataset/images/59041171_106970752.jpg 情笼罩在他们满是沧桑'
# '/media/chenjun/ed/23_OCR/Synthetic_Chinese_String_Dataset/Synthetic_Chinese_String_Dataset/images/50843500_2726670787 项链付出了十年的苦役']

"""a=b'20455828_2605100732.jpg 263 82 29 56 35 435 890 293 126 129\r\n'
print(a.decode('utf-8').strip())
20455828_2605100732.jpg 263 82 29 56 35 435 890 293 126 129"""

def get_dataset(opt):
    TRAIN_INFO=get_label(opt.TRAIN_DIR)
    TEST_INFO=get_label(opt.TEST_DIR)
    UNIQUE_CHAR=set(',')

    for label in np.hstack((TRAIN_INFO,TEST_INFO)):#在水平方向上平铺
        global label_list
        try:
            """a = '20455828_2605100732.jpg 项链付出了十年的苦役'
            label_list = list(a.split()[1])
            print(label_list)
            UNIQUE_CHAR = set(',')"""
            label_list=list(label.split()[1])
            #['项', '链', '付', '出', '了', '十', '年', '的', '苦', '役']
        except:
            print("NO CHARACTER!")
        for l in label_list:
            if not l in UNIQUE_CHAR:
                UNIQUE_CHAR.add(l)
                #如果在UNIQUE_CHAR里面没有出现过的话，就加入进去

    #write the unique char to the file
    with open('/home/std2021/hejiabang/OCR/Attention/unique_char.txt','w',encoding='utf-8') as f:
        for char in sorted(list(UNIQUE_CHAR)):
            f.write(char.strip()+'\n')

    char_to_index={x:y for x,y in zip(
        sorted(list(UNIQUE_CHAR))+['eof'],range(len(UNIQUE_CHAR)+1)
        #，我 是 戳 欻 三 大....eof
        #0,1,2,3,4,5,6....len+1
    )}

    index_to_char={y:x for x,y in zip(
        sorted(list(UNIQUE_CHAR))+[' '],[str(_) for _ in range(len(UNIQUE_CHAR)+1)]
    )}

    #write index_to_char into json file
    #{"0": "!", "1": "\"", "2": "#", "3": "$", "4": "%",...}
    with open('/home/std2021/hejiabang/OCR/Attention/index_to_char.json','w',encoding='utf-8') as f:
        json.dump(index_to_char,f)

    n_class=len(UNIQUE_CHAR)+1

    train_dataset=CNDATA(TRAIN_INFO,img_transforms=cn_transform(opt.input_h,opt.input_w),
                         label_transforms=label_transforms(char_to_index,n_class,opt.max_seq_len))

    test_dataset=CNDATA(TEST_INFO,img_transforms=cn_transform(opt.input_h,opt.input_w),
                        label_transforms=label_transforms(char_to_index,n_class,opt.max_seq_len))

    train_data=DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)

    test_data=DataLoader(test_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)

    return train_data,test_data,char_to_index,index_to_char,n_class

"""a=['项', '链', '付', '出', '了', '十', '年', '的', '苦', '役']
st=set('.')
for ch in a :
    st.add(ch)
print(st)
#{'苦', '役', '项', '链', '年', '出', '.', '的', '十', '了', '付'}

char_to_index={x:y for x,y in zip(
        sorted(list(st))+['eof'],range(len(st)+1)
        #，我 是 戳 欻 三 大....eof
        #0,1,2,3,4,5,6....len+1
    )}

pprint(char_to_index)
#{'.': 0, '了': 1, '付': 2, '出': 3, '十': 4, '年': 5, '役': 6, '的': 7, '苦': 8, '链': 9, '项': 10, 'eof': 11}
index_to_char={y:x for x,y in zip(
        sorted(list(st))+[' '],[str(_) for _ in range(len(st)+1)]
    )}
print(index_to_char)
#{'0': '.', '1': '了', '2': '付', '3': '出', '4': '十', '5': '年', '6': '役', '7': '的', '8': '苦', '9': '链', '10': '项', '11': ' '}

with open('../Attention_OCR/index_to_char.json', 'w', encoding='utf-8') as f:
    json.dump(index_to_char, f)"""

"""返回类型 label Tensor
[[False  True False False]
 [False False False  True]
 [False False  True False]
 [ True False False False]]"""