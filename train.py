import torch
import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from thop import profile  # 用于计算FLOPs
from WRes_Unet import WRes_UNet
from data_loader_split import *
from metrics import indicators
from torch.nn import functional as F

# 初始化参数和路径
data_name="ICH"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = data_name+'/params/osteosarcoma.pth'
log_path = (data_name+"/logs")
Train_save_path = data_name+"/train_image"
Test_save_path = data_name+"/test_image"
csv_savefile = data_name+"/results.csv"
t_num = 5
v_num = 1
cishu=1
writer = SummaryWriter(log_path)

# 一次性指标保存到txt
def save_model_info(model, input_size=(1, 1, 512, 512), txt_file=data_name+"/model_info.txt"):
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    with open(txt_file, 'w') as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"FLOPs: {flops}\n")


# 模型、优化器和损失函数
net = WRes_UNet(1, 1).to(device)
opt = optim.Adam(net.parameters(), lr=0.001)
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

loss_fun = BCEDiceLoss()#FocalLoss(alpha=0.75, gamma=2.0, reduction='mean') #nn.BCELoss()

# 保存多轮次指标到csv文件

def save_metrics_to_csv(metrics, epoch, csv_file=csv_savefile):
    # 每个 metric 添加 epoch 信息
    for metric in metrics:
        metric['epoch'] = epoch

    # 转换为 DataFrame
    df = pd.DataFrame(metrics)

    # 判断文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False, mode='w', header=True)
    else:
        df.to_csv(csv_file, index=False, mode='a', header=False)


# 主训练和验证函数
def train_and_evaluate(num_epochs, train_loader, test_loader):
    save_model_info(net)  # 保存参数量和FLOPs
    metrics = []  # 用于存储每个epoch的指标
    # 训练模式
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')
    for epoch in range(num_epochs):
        net.train()
        total_train_step, train_loss_total = 0, 0.0
        train_metrics = []

        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}") as train_pbar:
            for (i,(images, labels)) in enumerate(train_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                train_loss = loss_fun(outputs, labels)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ ,f1= indicators(outputs, labels)
                train_metrics.append({
                    'train_iou': iou_,
                    'train_dice': dice_,
                    'train_loss': train_loss.item(),
                    'train_f1': f1,
                    'train_hd95': hd95_,
                    'train_recall': recall_,
                    'train_specificity': specificity_,
                    'train_precision':precision_
                })

                _image = images[0]
                _segment_image = labels[0]
                _out_image = outputs[0]
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                if i % t_num == 0:
                    save_image(img, f'{Train_save_path}/{cishu}_{epoch}_{i}.png')
                total_train_step += 1
                train_loss_total += train_loss.item()

                # TensorBoard记录
                writer.add_scalar("train_loss", train_loss.item(), total_train_step)
                writer.add_scalar("train_iou", iou_, total_train_step)
                writer.add_scalar("train_dice", dice_, total_train_step)
                writer.add_scalar("train_recall", recall_, total_train_step)
                writer.add_scalar("train_specificity", specificity_, total_train_step)
                writer.add_scalar("train_precision", precision_, total_train_step)
                writer.add_scalar("train_f1", f1, total_train_step)
                writer.add_scalar("train_hd95", hd95_, total_train_step)

                train_pbar.set_postfix({
                    'Loss': train_loss.item(),
                    'specificity': specificity_,
                    'Precision': precision_,
                    'Recall': recall_,
                    'test_dice': dice_,
                    'IoU': iou_
                })

        # 测试模式
        net.eval()
        test_metrics = []
        with torch.no_grad(), tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{num_epochs}") as test_pbar:
            for (i,(images, labels)) in enumerate(test_pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                test_loss = loss_fun(outputs, labels)
                iou_, dice_, hd_, hd95_, recall_, specificity_, precision_,f1 = indicators(outputs, labels)
                test_metrics.append({
                    'test_iou': iou_,
                    'test_dice': dice_,
                    'test_loss': test_loss.item(),
                    'test_f1': f1,
                    'test_hd95': hd95_,
                    'test_recall': recall_,
                    'test_specificity': specificity_,
                    'test_precision': precision_
                })

                test_pbar.set_postfix({
                    'Loss': test_loss.item(),
                    'specificity': specificity_,
                    'Precision': precision_,
                    'Recall': recall_,
                    'test_dice': dice_,
                    'IoU': iou_
                })
                _image = images[0]
                _segment_image = labels[0]
                _out_image = outputs[0]
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                if i % v_num == 0:
                    save_image(img, f'{Test_save_path}/{cishu}_{epoch}_{i}.png')


        # 每个epoch的平均结果保存到metrics
        train_avg = pd.DataFrame(train_metrics).mean().to_dict()
        test_avg = pd.DataFrame(test_metrics).mean().to_dict()
        epoch_metrics = {'epoch': epoch + 1,**train_avg, **test_avg}
        metrics_list = [epoch_metrics]
        metrics.append(epoch_metrics)

        # 打印每个epoch的训练和测试平均指标
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Metrics: {train_avg}")
        print(f"Epoch {epoch + 1}/{num_epochs} - Testing Metrics: {test_avg}")
        save_metrics_to_csv(metrics_list, epoch=epoch+1)
        # 保存模型权重
        torch.save(net.state_dict(), weight_path)
        print(f"Model saved at epoch {epoch + 1}.")




if __name__ == '__main__':
    root_path = data_name+"/data"
    image_path = "imgs"
    label_path = "masks"
    batch_size = 1

    # 首先执行数据划分
    #split_data(root_path, label_path, train_txt=data_name+"/information_txt/train.txt", test_txt=data_name+"/information_txt/test.txt", test_split=0.3)

    # 创建数据加载器
    train_loader, test_loader = create_train_test_loaders(root_path, image_path, label_path, train_txt=data_name+"/information_txt/train.txt", test_txt=data_name+"/information_txt/test.txt", batch_size=8,num_workers=10)

    # 打印训练和测试集大小
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    train_and_evaluate(200, train_loader, test_loader)