import torch

class Attention_loss():
    #[batch,5586,10],[1,5586,10]
    #eg:[3,4,3]与[1,4,3],4个映射，3个字符长度
    def __call__(self,outputs,labels, eof_weight=1):
        #此处不要打成cell
        blank_index=(labels.max(1)[1]==labels.shape[1]-1)
        char_index=~blank_index
        cl=labels.max(1)[1][char_index]
        co=outputs.transpose(1,2)[torch.unbind(char_index.nonzero(),dim=1)]#[10,5586]
        cross_entropy=torch.nn.CrossEntropyLoss()
        loss=cross_entropy(co,cl)

        return loss

"""labels=torch.FloatTensor(
[[[1 ,0 ,0 ,0],
 [0 ,1 ,0 ,0],
 [0 ,0 ,0 ,0 ],
 [0 ,0 ,0 ,1 ],
 [0 ,0 ,1 ,0 ]]])

outputs=torch.FloatTensor(
[[[0.0,0.2 ,1.0 ,0.1 ],
 [0.1 ,1.0 ,0.1 ,0.1 ],
 [1.0 ,0.2 ,0.2 ,0.2 ],
 [0.1 ,0.2 ,0.1 ,1.0 ],
 [0.1 ,0.2 ,0.2 ,0.1 ]]])

outputs2=torch.FloatTensor(
[[[1.0,0.2 ,1.0 ,0.1 ],
 [0.1 ,1.0 ,0.1 ,0.1 ],
 [0.2 ,0.2 ,0.2 ,0.2 ],
 [0.1 ,0.2 ,0.1 ,1.0 ],
 [0.1 ,0.2 ,0.2 ,0.1 ]]])"""

"""a=tuple(([0, 0, 0], [0, 1, 3]))
print(outputs[a])
tensor([[0.0000, 0.2000, 1.0000, 0.1000],
        [0.1000, 1.0000, 0.1000, 0.1000],
        [0.1000, 0.2000, 0.1000, 1.0000]])"""

#blank_idx=(labels.max(1)[1]==labels.shape[1]-1)
#print(blank_idx)
#tensor([[False, False, False, False]])
#char_index=~blank_idx
#print(char_index)
#tensor([[True, True, True, True]])
#cl=labels.max(1)[1][char_index]
#print(cl)
#print(cl)
#tensor([0, 1, 0, 3])
#co=outputs.transpose(1,2)[torch.unbind(char_index.nonzero(),dim=1)]
#print(torch.unbind(char_index.nonzero(),dim=1))
#(tensor([0, 0, 0]), tensor([0, 1, 3]))
#print(char_index.nonzero())
#tensor([[1.0000, 0.1000, 0.1000, 0.1000, 0.1000],
#        [0.2000, 1.0000, 0.2000, 0.2000, 0.2000],
#        [1.0000, 0.1000, 0.2000, 0.1000, 0.2000],
#        [0.1000, 0.1000, 0.2000, 1.0000, 0.1000]])
#print(co)
#cross_entropy=torch.nn.CrossEntropyLoss()
#loss=cross_entropy(co,cl)
#print(loss)
#tensor(0.9934)
#当调整1.0的位置时出现较大的loss tensor(1.2397)"""

"""#损失函数实验
import torch
import torch.nn as nn

x_input=torch.randn(3,3)
print("x_input:\n",x_input)

#设置输出具体值
y_target=torch.tensor([1,2,0])

#计算输入softmax，此时可以看到每一行加到一起的结果都是1
softmax_func=nn.Softmax(dim=1)
soft_output=softmax_func(x_input)
print('soft_output:\n',soft_output)

#在softmax的基础上取log
log_output=torch.log(soft_output)
print('log_output:\n',log_output)

#对比softmax和log结合，与nn.LongSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的
logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)

#pytorch中关于NLLLoss的默认参数配置为：reduction=True，size_average=True
nllloss_func=nn.NLLLoss()
nllloss_output=nllloss_func(logsoftmax_output,y_target)
print('nllloss_output:\n',nllloss_output)

#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()答案时一致的
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crosstropyloss_output:\n',crossentropyloss_output)"""

"""
x_input:
 tensor([[-0.8544, -0.0921,  1.6346],
        [-1.5638,  0.0843,  1.0429],
        [ 0.6456,  0.0423,  1.0228]])
soft_output:
 tensor([[0.0658, 0.1411, 0.7931],
        [0.0506, 0.2631, 0.6862],
        [0.3328, 0.1820, 0.4852]])
log_output:
 tensor([[-2.7208, -1.9585, -0.2318],
        [-2.9832, -1.3351, -0.3765],
        [-1.1003, -1.7036, -0.7232]])
logsoftmax_output:
 tensor([[-2.7208, -1.9585, -0.2318],
        [-2.9832, -1.3351, -0.3765],
        [-1.1003, -1.7036, -0.7232]])
nllloss_output:
 tensor(1.1451)
crosstropyloss_output:
 tensor(1.1451)
"""

"""x=torch.FloatTensor([[[1.0,0.5],[0.5,0.2],[0.5,0.6]]])

print(x[([0,0],[1,0])])
#tensor([[0.5000, 0.2000],
#        [1.0000, 0.5000]])

x=torch.FloatTensor([[1.0,0.5],[0.5,0.2],[0.5,0.6]])

print(x[([0,2],[1,0])])
#tensor([0.5000, 1.0000])"""
