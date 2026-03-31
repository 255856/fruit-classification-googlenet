import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import GoogLeNet,Inception
from PIL import Image

def test_data_process():
    # 数据集路径
    ROOT_TRAIN = r'data\test'
    # 归一化
    normalize = transforms.Normalize(mean=[0.22890568, 0.19639583, 0.1433638], std=[0.09950783, 0.07997292, 0.06596899])

    # 定义数据集处理方法变量，图片大小，ToTensor格式
    test_treansform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(root=ROOT_TRAIN, transform=test_treansform)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0)#进程

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #初始化参数
    test_corrects = 0.0#精确度
    test_num = 0#数量

    with torch.no_grad():#推理只进行前向传播，不计算梯度，节省内存
        for test_data_x, test_data_y in test_dataloader:#一批中只有一张样本
            #将数据放入GPU
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            #将模型设置为验证模式
            model.eval()
            #前向传播过程，输入测试集数据，输出对应每个样本预测
            output = model(test_data_x)
            #获取最大概率标签，查找每一行最大值对应标签
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确准确度+1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            #测试样本累加
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print('测试准确率: {}'.format(test_acc))

if __name__ == '__main__':
    model = GoogLeNet(Inception)
    #加载训练好的模型
    model.load_state_dict(torch.load('fruits_best_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    classes = ['apple', 'banana', 'grape', 'orange', 'pear']

    # test_dataloader = test_data_process()
    # test_model_process(model, test_dataloader)
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         model.eval()
    #         output = model(b_x)
    #         # 获取最大概率标签，查找每一行最大值对应标签
    #         pre_lab = torch.argmax(output, dim=1)
    #
    #         #
    #         result = pre_lab.item()
    #         lable = b_y.item()
    #         print("预测值：",classes[result],"------","真实值：",classes[lable])

    """单一照片识别"""
    image = Image.open('apple.png')
    normalize = transforms.Normalize(mean=[0.22890568, 0.19639583, 0.1433638], std=[0.09950783, 0.07997292, 0.06596899])
    # 定义数据集处理方法变量，图片大小，ToTensor格式
    test_treansform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_treansform(image)
    #添加批次维度
    image = image.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：", classes[result])