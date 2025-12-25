import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import tqdm

# ================= 配置参数 =================
CONFIG = {
    'img_size': 256,        # ResNet 标准输入通常是 224 或 256
    'batch_size': 32,       # 4GB 显存下，ResNet-18 跑 Batch 32 比较稳妥
    'lr': 0.001,
    'epochs': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_dir': r'D:\table\ml\project\dataset\train', 
    'test_dir': r'D:\table\ml\project\dataset\test',
    'pos_weight': 9.0,      # 类别不平衡权重
    'num_workers': 4        # 多进程加载
}

# 开启 CUDNN 加速
torch.backends.cudnn.benchmark = True 

# ================= 数据集类 (保持不变) =================
class GlassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, 'img')
        self.txt_dir = os.path.join(root_dir, 'txt')
        self.img_paths = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        txt_path = os.path.join(self.txt_dir, filename.replace('.png', '.txt'))
        label = 1.0 if os.path.exists(txt_path) else 0.0
        
        image = Image.open(img_path).convert('L') # 保持单通道灰度输入
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ================= 模型架构：ResNet-18 (手动实现版) =================

# 1. 定义残差块 (Residual Block)
# 这是深层网络不退化的核心： x -> Conv -> ReLU -> Conv -> (+ x) -> ReLU
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # 用于调整维度，以便 x 能和 conv(x) 相加

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # 【关键】跳跃连接
        out = self.relu(out)

        return out

# 2. 定义主网络 ResNet
class ResNetGlass(nn.Module):
    def __init__(self):
        super(ResNetGlass, self).__init__()
        
        # 初始层 (Stem): 快速降维
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个主要的层 (Layer)，每个 Layer 包含 2 个 ResidualBlock
        # 结构：[64通道] -> [128通道] -> [256通道] -> [512通道]
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # GAP
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        # 如果输入输出通道不一致，或者步长不为1，需要对残差边(identity)做卷积来匹配维度
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # 此时特征已经非常高级且抽象

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ================= 训练与评估函数 =================
def train_model():
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10), # 增加一点旋转增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_ds = GlassDataset(CONFIG['train_dir'], transform=train_transform)
    test_ds = GlassDataset(CONFIG['test_dir'], transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = ResNetGlass().to(CONFIG['device'])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG['pos_weight']]).to(CONFIG['device']))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 混合精度 Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 学习率调整策略 (Warmup + Cosine 或者 ReduceLROnPlateau)
    # 这里用 ReduceLROnPlateau：当 Loss 不降时，自动减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    print(f"开始训练 ResNet-18, 设备: {CONFIG['device']} | AMP: On")
    
    best_f1 = 0
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for imgs, labels in tqdm.tqdm(train_loader):
            imgs = imgs.to(CONFIG['device'], non_blocking=True)
            labels = labels.to(CONFIG['device'], non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        # 验证
        model.eval()
        all_labels = []
        all_preds = []
        test_loss = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(CONFIG['device'], non_blocking=True)
                labels = labels.to(CONFIG['device'], non_blocking=True).unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    t_loss = criterion(outputs, labels)
                
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                test_loss += t_loss.item()

        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        all_probs = torch.sigmoid(torch.tensor(all_preds)).numpy().flatten() # 注意这里要用概率值
        
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        p = precision_score(all_labels, all_preds, zero_division=0)
        r = recall_score(all_labels, all_preds, zero_division=0)
        
        # 更新学习率 (根据 F1 分数)
        scheduler.step(f1)

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"F1: {f1:.4f} (P: {p:.4f}, R: {r:.4f}) | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        best_thr = 0.5
        best_f1 = 0
        best_metrics = (0, 0) # (P, R)

        # 搜索阈值
        for thr in np.arange(0.3, 0.95, 0.05):
            y_pred_thr = (all_probs > thr).astype(int)
            f1_t = f1_score(all_labels, y_pred_thr, zero_division=0)
            
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thr = thr
                p_t = precision_score(all_labels, y_pred_thr, zero_division=0)
                r_t = recall_score(all_labels, y_pred_thr, zero_division=0)
                best_metrics = (p_t, r_t)

        print(f"✅ 最佳阈值: {best_thr:.2f} | F1: {best_f1:.4f} (P: {best_metrics[0]:.4f}, R: {best_metrics[1]:.4f})")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_resnet_glass.pth')
            print("--> 👑 保持最佳模型")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support() 
    train_model()