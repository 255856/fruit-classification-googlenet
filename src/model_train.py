import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms#处理数据
import torch.utils.data as Data
import copy
import time
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')#切换matplotlib后端
import matplotlib.pyplot as plt
from model import GoogLeNet,Inception

def train_val_data_process():
    #数据集路径
    ROOT_TRAIN =  r'data\train'
    #归一化
    normalize = transforms.Normalize(mean=[0.22890568, 0.19639583, 0.1433638], std=[0.09950783, 0.07997292, 0.06596899])

    #定义数据集处理方法变量，图片大小，ToTensor格式
    train_treansform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),normalize])
    #加载数据集
    train_data = ImageFolder(root=ROOT_TRAIN, transform=train_treansform)


    #划分训练集，验证集
    train_data,val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])#80%训练集

    train_dataloader = Data.DataLoader(dataset=train_data,
                                        batch_size=64,
                                        shuffle=True,#是否打乱
                                        num_workers=2)#进程

    val_dataloader = Data.DataLoader(dataset=val_data,
                                      batch_size=64,
                                      shuffle=True,
                                      num_workers=2)

    return train_dataloader,val_dataloader

def train_model_process(model, train_dataloader, val_dataloader ,num_epochs):#模型，训练集，验证集，训练轮次
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU还是CPU

    #使用Adam优化器，学习率0.001（Adam类似梯度下降法
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    #损失函数
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数（分类），均方差（回归）
    #将模型放入GPU
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高精确度
    best_acc = 0.0
    #训练集损失列表，训练集准确度列表，验证集损失列表，验证集准确度列表
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = [], [], [], []
    #当前时间
    since = time.time()

    for epoch in range(num_epochs):
        #打印轮次
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 50)

        #初始化参数
        #训练集损失函数
        train_loss = 0.0
        #训练集精确度
        train_corrects = 0.0
        #验证集损失函数
        val_loss = 0.0
        # 验证集精确度
        val_corrects = 0.0
        #训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        #对每一个mini-batch训练计算
        for step, (b_x, b_y) in enumerate(train_dataloader):#128*224*224*1的数据
            #将特征和标签放入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为训练模式
            model.train()
            #前向传播,输出10个值
            output = model(b_x)
            #查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output, b_y)
            #将梯度置为0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #根据网络反向传播的梯度信息来更新网络参数
            optimizer.step()
            #对损失函数进行累加,一个批次的loss累加
            train_loss += loss.item() * b_x.size(0)
            #如果预测正确准确度+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            #当前用于训练样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #开设置模型为评估模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确准确度+1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练样本数量
            val_num += b_x.size(0)

        #每个批次的平均loss值不同，所以先要求loss的累加和，再除以总样本数，得到一个epoch的平均loss值
        #计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印列表最后一个数据，即为当轮数据
        print('{} Train_Loss: {:.4f} Train_Acc: {:.4f}'.format(epoch+1, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val_Loss: {:.4f} Val_Acc: {:.4f}'.format(epoch+1, val_loss_all[-1], val_acc_all[-1]))

        #寻找高准确度的权重
        if val_acc_all[-1] > best_acc:
            #将最高准确度赋值给best_acc
            best_acc = val_acc_all[-1]
            #保存最优模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        #训练耗费时间
        time_use = time.time() - since
        print('训练和验证耗费时间： {:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))
        print('-' * 50)

    #选择最优参数
    #加载最高准确率下的参数
    torch.save(best_model_wts, 'fruits_best_model.pth')

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_loss_all":val_loss_all,
                                       "val_acc_all":val_acc_all})#键：值
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)#一行两列第一张图
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)#一行两列第一张图
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()

if __name__ == '__main__':
    #将模型实例化
    model = GoogLeNet(Inception)
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, 50)
    matplot_acc_loss(train_process)