import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

# 定义图像填充函数，使图像保持指定大小
def keep_image_size_open(path, size=(512, 512)):
    img = Image.open(path).convert("L")  # 转为灰度模式
    temp = max(img.size)
    mask = Image.new('L', (temp, temp), 0)  # 创建灰度背景
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_image_rgb_size_open(path,size=(512,512)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask

def keep_mask_image_size_open(path, size=(512, 512), background_color=0, foreground_color=255):
    # 打开图像并转换为灰度模式
    img = Image.open(path).convert("L")
    img = np.array(img)  # 转换为 NumPy 数组，方便处理

    # 将前景（非零像素）设置为白色，背景设置为黑色
    img[img > 0] = foreground_color  # 非零像素（前景）设为白色
    img[img == 0] = background_color  # 零像素（背景）设为黑色

    # 创建PIL图像对象
    img = Image.fromarray(img)

    # 调整图像大小
    img = img.resize(size)

    return img

# 图像预处理步骤
transform = transforms.Compose([
    transforms.ToTensor()
])

# 划分数据集
import os
import random


# 获取数据集标签，并按比例划分为训练集和测试集
def split_data(root_path, label_path, train_txt="train.txt", test_txt="test.txt", test_split=0.3):
    # 获取标签目录下所有文件名
    label_files = os.listdir(os.path.join(root_path, label_path))

    # 随机打乱标签
    random.shuffle(label_files)

    # 计算训练集和测试集大小
    total_count = len(label_files)
    test_count = int(total_count * test_split)
    train_count = total_count - test_count

    # 划分数据
    train_files = label_files[:train_count]
    test_files = label_files[train_count:]

    # 保存训练集和测试集文件路径
    with open(train_txt, 'w') as train_file:
        for item in train_files:
            train_file.write(item + "\n")

    with open(test_txt, 'w') as test_file:
        for item in test_files:
            test_file.write(item + "\n")

    print(f"Data split complete. {len(train_files)} for training, {len(test_files)} for testing.")


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, root_path, image_path, label_path, txt_file, image_size=(512, 512)):
        self.root_path = root_path
        self.label_path = label_path
        self.image_path = image_path
        self.image_size = image_size

        # 读取txt文件中的路径
        with open(txt_file, 'r') as file:
            self.file_names = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # 获取图像和标签路径
        segment_name = self.file_names[index]
        segment_path = os.path.join(self.root_path, self.label_path, segment_name)
        img_path = os.path.join(self.root_path, self.image_path, segment_name.replace("", ""))

        # 加载并调整图像尺寸
        image = keep_image_size_open(img_path, self.image_size)
        segment_image = keep_mask_image_size_open(segment_path, self.image_size)

        # 转换为Tensor
        return transform(image), transform(segment_image)


# 创建训练集和测试集的DataLoader
def create_train_test_loaders(root_path, image_path, label_path, train_txt="train.txt", test_txt="test.txt", batch_size=1,num_workers=20):
    # 使用划分好的 txt 文件
    train_dataset = MyDataset(root_path, image_path, label_path, txt_file=train_txt)
    test_dataset = MyDataset(root_path, image_path, label_path, txt_file=test_txt)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=True)

    return train_loader, test_loader

# 测试数据加载器
if __name__ == '__main__':
    root_path = "ICH"
    image_path = "imgs"
    label_path = "masks"
    batch_size = 2

    # 首先执行数据划分
    split_data(root_path+'/data', label_path, train_txt=root_path+"/information_txt/train.txt", test_txt=root_path+"/information_txt/test.txt", test_split=0.3)

    # 创建数据加载器
    train_loader, test_loader = create_train_test_loaders(root_path+'/data', image_path, label_path, train_txt=root_path+"/information_txt/train.txt", test_txt=root_path+"/information_txt/test.txt", batch_size=batch_size)

    # 打印训练和测试集大小
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    # 测试一个批次的数据
    train_images, train_labels = next(iter(train_loader))
    print(f"Train batch - images shape: {train_images.shape}, labels shape: {train_labels.shape}")
