# 水果分类系统 - GoogLeNet

基于GoogLeNet (Inception) 深度学习模型的水果图像分类系统。

## 项目简介

本项目使用GoogLeNet神经网络架构对水果图像进行分类。GoogLeNet通过Inception模块提高了网络宽度，在保持计算效率的同时提升了分类准确率。

## 模型架构

- **基础网络**: GoogLeNet (Inception v1)
- **输入尺寸**: 224×224×3
- **输出类别**: 5类水果
- **损失函数**: 交叉熵损失
- **优化器**: Adam (学习率=0.001)

### Inception模块结构
- 1×1卷积层
- 1×1卷积 + 3×3卷积
- 1×1卷积 + 5×5卷积
- 3×3最大池化 + 1×1卷积

## 环境要求

```bash
Python 3.8+
PyTorch 1.9+
torchvision
Pillow
numpy
pandas
matplotlib
```
## 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/fruit-classification-googlenet.git
cd fruit-classification-googlenet
```
### 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. 数据集准备
```bash
将水果图像按类别放入 fruits/ 目录：

text
fruits/
├── apple/
│   ├── image1.jpg
│   └── ...
├── banana/
├── orange/
├── grape/
└── pear/
```
### 4. 数据预处理
```bash
划分训练集和测试集：
python src/data_partitioning.py

计算数据集均值和标准差（用于归一化）：
python src/mean_std.py
```
### 5. 训练模型
```bash
python src/model_train.py
训练完成后，最佳模型将保存为 fruits_best_model.pth
```
### 6. 测试模型
```bash
python src/model_test.py
```
## 项目结构
```bash
├── src/                    # 源代码目录  
│   ├── data_partitioning.py   # 数据集划分  
│   ├── mean_std.py            # 计算均值和标准差  
│   ├── model.py               # GoogLeNet模型定义  
│   ├── model_train.py         # 训练脚本  
│   ├──model_test.py          # 测试脚本
│   ├──apple.png              #测试用例
│   └── fruits_best_model.pth   #训练好的模型参数
├── data/                   # 处理后的数据集  
│   ├── train/              # 训练集  
│   └── test/               # 测试集   
├── fruits/                 # 原始数据（需自行准备）  
├── requirements.txt        # 项目依赖  
└── README.md              # 项目说明  
```
## 训练过程
```bash
训练过程中会显示：
每个epoch的训练损失和准确率
每个epoch的验证损失和准确率

训练和验证耗时
训练结束后会生成：
损失和准确率变化曲线图
最佳模型权重文件
```
### 参数配置
```bash
批量大小	> 64  
训练轮数	> 50  
学习率	> 0.001  
训练/验证集比例	> 8:2  
图片尺寸	> 224×224 
类别数 > model.py中nn.Linear(1024,num)
类别 > classes = []
``` 
### 结果
训练完成后，模型会在测试集上输出分类准确率，并可以对单张图片进行预测。
