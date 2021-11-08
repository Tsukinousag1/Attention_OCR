import logging
import torch
from meter import AverageValueMerter
import shutil
import os

#创建目录文件
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


#Save the model
def save_checkpoint(state,is_best,save_dir,filename='checkpoint.pt'):
    fpath='_'.join((str(state['epoch']),filename))
    fpath=os.path.join(save_dir,fpath)
    #合并路径
    make_dir(save_dir)
    #先建立save_dir文件目录
    torch.save(state,fpath)
    #保存参数
    if is_best:
        #复制文件从src到dst，其中fpath必须是完整的目标文件名
        shutil.copy(fpath,os.path.join(save_dir,'model_best.pt'))

#fpath='_'.join(('100','hjb'))
#print(fpath)
class Train_Engine(object):
    def __init__(self,net):
        self.net=net
        self.loss=AverageValueMerter()
        self.seq_acc=AverageValueMerter()
        self.char_acc=AverageValueMerter()

    def fit(self,index_to_char,train_data,test_data,optimizer,criterion,epochs=300,print_interval=100,eval_step=1,save_step=1,save_dir='checkpoint',use_gpu=True):
        best_test_acc=0.0
        for epoch in range(0,epochs):
            self.loss.reset()
            self.seq_acc.reset()
            self.char_acc.reset()
            self.net.train()

            for i,data in enumerate(train_data):
                self.net.train()
#运行pytorch时，训练很正常，但是如果切换到eval()模式之后再继续训练， 发现报错：
#RuntimeError: cudnn RNN backward can only be called in training mode
                imgs,labels=data
                if use_gpu:
                    labels=labels.cuda()
                    #torch.Size([1,5586,10]) map中5586个字符
                    imgs=imgs.cuda()

                outputs=self.net(imgs,labels)
                #[batch_size,5586,10]

                loss=criterion(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #####
                self.loss.add(loss.item())

                # [batch_size,5586,10],torch.prob(,dim=1)
                seq_acc=(outputs.max(1)[1].long()==labels.max(1)[1].long()).prod(1).float().mean()
                char_acc=(outputs.max(1)[1].long()==labels.max(1)[1].long()).float().mean()

                self.seq_acc.add(seq_acc.item())
                self.char_acc.add(char_acc.item())

                loss_mean = self.loss.value()
                seq_acc_mean = self.seq_acc.value()
                char_acc_mean = self.char_acc.value()

                if print_interval and (i+1)%(5*print_interval)==0:
                    pred_text=''.join([index_to_char[str(_)] for _ in outputs[0].max(0)[1].cpu().numpy().tolist()])
                #a=torch.Tensor([[[1,2,3],[1,4,5],[7,5,3],[1,2,5]]])
                #a[0].max(0),a[0].max(0)[1]
                #(torch.return_types.max(
                #    values=tensor([7., 5., 5.]),
                #    indices=tensor([2, 2, 1])),
                # tensor([2, 2, 1]))
                #也就是把10个输出里面填入5586里最有可能的10个index
                    label_text=''.join([index_to_char[str(_)] for _ in labels[0].max(0)[1].cpu().numpy().tolist()])
                    logging.info('%-11s ==> gt: %-11s' % (pred_text,label_text))

                if print_interval and (i+1)%print_interval==0:

                    logging.info('Epoch: %d\tBatch: %d\tloss=%f\tseq_acc=%f\tchar_acc=%f'
                                 % (epoch,i+1,loss_mean,seq_acc_mean,char_acc_mean))

                #正常情况下，输出以下信息
                logging.info('Epoch: %d\ttraining: loss=%f\tepoch_seq_acc=%f\tepoch_char_acc=%f'
                             % (epoch,loss_mean,seq_acc_mean,char_acc_mean))


                is_best=True
                if test_data is not None and (epoch+1)%eval_step==0:
                    test_seq_acc,test_char_acc=self.val(test_data)
                    logging.info('----->> Epoch: %d\ttest_seq_acc=%f\ttest_char_acc=%f' % (epoch,test_seq_acc,test_char_acc))
                    is_best=test_seq_acc > best_test_acc
                    if is_best:
                        best_test_acc=test_seq_acc

                    state_dict=self.net.module.state_dict()
                    if not (epoch+1)% save_step :
                        save_checkpoint({
                            'state_dict':state_dict,
                            'epoch':epoch+1
                        },is_best=is_best,save_dir=save_dir)

        print('Finished\n')

    #val the model
    def val(self,test_data):
        global seq_acc, char_acc
        self.net.eval()
        for idx,data in enumerate(test_data):
            imgs,labels=data
            #print(imgs.shape,labels.shape)
            labels=labels.cuda()
            outputs=self.net(imgs)

            char_acc=(outputs.max(1)[1].long()==labels.max(1)[1].long()).float().mean()
            seq_acc=(outputs.max(1)[1].long()==labels.max(1)[1].long()).prod(1).float().mean()

        return seq_acc,char_acc



