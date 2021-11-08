import torch
import torch.nn as nn
from torchvision.models import vgg

DECODER_INPUT_SIZE=256

DECODER_HIDDEN_SIZE=256
ENCODER_HIDDEN_SIZE=256

DECODER_OUTPUT_FC=256
DECODER_OUTPUT_FRAME=10

V_FC=50
V_SIZE=50

class VGG11Base(nn.Module):
    def __init__(self):
        super(VGG11Base, self).__init__()
        vgg11=vgg.vgg11_bn(pretrained=True)
        vgg11.features[14]=nn.MaxPool2d((2,2),(2,1),(0,1))
        #batch [256,4,26]
        vgg11.features[21]=nn.MaxPool2d((2,2),(2,1),(0,1))
        #batch [512,2,27]
        vgg11.features[22]=nn.Conv2d(
            512,512,kernel_size=(2,2),stride=(2,1),padding=(0,0))
        #batch [512,1,26]
        self.vgg11_base=vgg11.features[:25]

    def forward(self,inputs):
        return self.vgg11_base(inputs)

    def out_channels(self):
        return self.vgg11_base[-3].out_channels# 512

class Attention_ocr(nn.Module):
    def __init__(self,use_gpu,NUM_CLASS):
        super(Attention_ocr, self).__init__()
        self.base_cnn=VGG11Base()
        self.NUM_CLASS=NUM_CLASS
        FEATURE_C=self.base_cnn.out_channels()#512
        self.lstm=nn.LSTM(input_size=FEATURE_C,hidden_size=DECODER_HIDDEN_SIZE,batch_first=True,bidirectional=True)
        #[512,256]
        self.rnn_cell=nn.GRUCell(input_size=DECODER_INPUT_SIZE,hidden_size=DECODER_HIDDEN_SIZE)
        #[256,256]
        self.layer_cx=nn.Linear(in_features=NUM_CLASS,out_features=DECODER_INPUT_SIZE)
        #[37,256]
        self.layer_ux=nn.Linear(in_features=FEATURE_C,out_features=DECODER_INPUT_SIZE)
        #[512,256]
        self.layer_so=nn.Linear(in_features=DECODER_HIDDEN_SIZE,out_features=DECODER_OUTPUT_FC)
        #[256,256]
        self.layer_uo=nn.Linear(in_features=FEATURE_C,out_features=DECODER_OUTPUT_FC)
        #[512,256]
        self.layer_oo=nn.Linear(in_features=DECODER_OUTPUT_FC,out_features=NUM_CLASS)
        #[256,37]
        self.layer_sa=nn.Linear(in_features=DECODER_HIDDEN_SIZE,out_features=V_FC)
        #[256,50]
        self.layer_fa=nn.Linear(in_features=DECODER_HIDDEN_SIZE*2,out_features=V_FC)
        #[512,50]
        self.layer_va=nn.Linear(in_features=V_FC,out_features=V_SIZE)
        #[50,50]
        self.layer_aa=nn.Linear(in_features=V_SIZE,out_features=1)
        #[50,1]
        self.use_gpu=use_gpu

    def forward(self,inputs,labels=None,return_alpha=False):
        if self.training:
            assert (labels is not None)
        batch_size=inputs.shape[0]#[3]
        #batch_size*c*(h*w)
        f=self.base_cnn(inputs)
        #torch.size[3,512,1,26]
        #=batch_size*seq_len*c
        #[3,512,26]->[3,26,512]
        f=f.view(batch_size,f.shape[1],-1).transpose(1,2)
        #batch_size*seq_len*(hidden_size*2):[3,26,512]
        f,_=self.lstm(f)#[3,26,512]
        #batch_size*(hidden_size*2)*seq_len
        f=f.transpose(1,2)#[3,512,26]
        c=torch.zeros(batch_size,self.NUM_CLASS)
        #[3,37]
        s=torch.zeros(batch_size,DECODER_HIDDEN_SIZE)
        #[3,256]
        if self.use_gpu:
            c,s=c.cuda(),s.cuda()
        outputs=[]
        alphas=[]
        for frame in range(DECODER_OUTPUT_FRAME):#10
            #f:[3,512,26] s:[3,256]
            alpha,u=self._get_alpha_u(f,s)
            ##alpha:[3,26] u:[3,512]
            alphas.append(alpha.view(batch_size,-1))
            #alphas:[[3,26],[3,26]...]
            x=self.layer_ux(u)+self.layer_cx(c)#3,256
            #x:[3,256] + [3,256]->[3,256]
            s=self.rnn_cell(x,s)#[batch_size,input_size] [batch_size,hidden_size]->[batch_size,hidden_size]
            #s:[3,256]
            o=self.layer_uo(u)+self.layer_so(s)
            #o:[3,256]+[3,256]->[3,256]
            o=self.layer_oo(nn.Tanh()(o))
            #o:torch.Size([3,37])
            outputs.append(o)
            #outputs:[[3,37],[3,37]...]

            #update c from o(evaluating) or ground truth(training)
            #更新c
            #训练模式
            if self.training:
                c=labels[:,:,frame]#ground truth对应frame的label
            # 验证模式
            else:
                c=nn.Softmax(dim=1)(o).detach()
                #不是训练模式，直接第一层就用softmax输出
                c=(c==c.max(1,keepdim=True)[0]).float()
                #torch.return_types.max(
                    #values=tensor([[3.],
                                   #[6.],
                                   #[4.]]),
                    #indices=tensor([[2],
                                   #[2],
                                   #[2]]))
        #[7,3,37]->[3,37,7],由10退化为7
        outputs=torch.stack(outputs,dim=-1) #torch.Size([3,37,10])
        if return_alpha:
            alphas=torch.stack(alphas,dim=-1)   #torch.Size([3,26,10])
            return outputs,alphas

        return outputs

    # torch.Size([3,37,10]) torch.Size([3,26,10])

    def _get_alpha_u(self,f,s):#[3,512,26] [3,256]
        a=self.layer_va(nn.Tanh()(
            #[3,26,512]->[3,26,50]          [3,256]->[3,1,50]
            self.layer_fa(f.transpose(1,2))+self.layer_sa(s).unsqueeze(1)))
        #[3,26,50]->[3,26,1]->[3,26]
        a=self.layer_aa(nn.Tanh()(a)).squeeze(-1)
        alpha=nn.Softmax(dim=1)(a)#[3,26]
        u=(f*alpha.unsqueeze(1)).sum(-1)#[3,512,26]*[3,1,26]->[3,512,26]->[3,512]
        #-1把所有行各自相加，-2把所有列各自相加
        return alpha,u
        #[3,26] [3,512]


"""if __name__ == '__main__':
    x = torch.randn(3,3,32,100)
    vgg_ = vgg.vgg11_bn(pretrained=True)
    vgg_v =VGG11Base()
    a=vgg_v(x)
    #print(a.shape)
    #torch.Size([3, 512, 1, 26])
    net=Attention_ocr(False,37)
    labels=torch.randn(3,37,10)
    output,alpha=net(x,labels,True)
    #print(output.shape,alpha.shape)
    #torch.Size([3, 37, 10]) torch.Size([3, 26, 10])"""
#[3 5586 10]

