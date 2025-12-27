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
    'img_size': 256,
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_dir': r'D:\table\ml\project\dataset_try\train', 
    'test_dir': r'D:\table\ml\project\dataset_try\test',
    'num_workers': 4
}

# 开启 CUDNN 加速
torch.backends.cudnn.benchmark = True 

# ================= 数据集类 =================
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
        
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ================= 模型架构：ResNet-18 =================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetGlass(nn.Module):
    def __init__(self):
        super(ResNetGlass, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
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
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ================= 训练与评估函数 =================
def train_model():
    # 1. 数据准备
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
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
    
    # 自动计算 pos_weight
    print("正在计算正负样本比例...")
    num_pos = sum(1 for _, label in train_ds if label == 1)
    num_neg = len(train_ds) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Pos: {num_pos}, Neg: {num_neg}, Calculated pos_weight: {pos_weight_val:.2f}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)

    # 2. 模型初始化
    model = ResNetGlass().to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(CONFIG['device']))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    print(f"开始训练 ResNet-18, 设备: {CONFIG['device']} | AMP: On")
    
    global_best_f1 = 0.0 # 全局最佳 F1
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        
        for imgs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False):
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

        # --- 验证阶段 ---
        model.eval()
        all_labels = []
        all_probs = [] # 【修复】这里只存概率值，不存 0/1
        test_loss = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(CONFIG['device'], non_blocking=True)
                labels = labels.to(CONFIG['device'], non_blocking=True).unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    t_loss = criterion(outputs, labels)
                
                # 【修复】获取 Sigmoid 后的概率值
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                test_loss += t_loss.item()

        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # 1. 计算默认阈值 0.5 下的指标
        default_preds = (all_probs > 0.5).astype(int)
        f1_def = f1_score(all_labels, default_preds, zero_division=0)
        p_def = precision_score(all_labels, default_preds, zero_division=0)
        r_def = recall_score(all_labels, default_preds, zero_division=0)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"F1(0.5): {f1_def:.4f} (P: {p_def:.4f}, R: {r_def:.4f}) | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 2. 搜索本 Epoch 的最佳阈值
        epoch_best_f1 = 0
        epoch_best_thr = 0.5
        
        for thr in np.arange(0.1, 0.95, 0.05):
            y_pred_thr = (all_probs > thr).astype(int)
            f1_t = f1_score(all_labels, y_pred_thr, zero_division=0)
            if f1_t > epoch_best_f1:
                epoch_best_f1 = f1_t
                epoch_best_thr = thr

        print(f"   >>> 本轮最佳阈值: {epoch_best_thr:.2f} | F1: {epoch_best_f1:.4f}")

        # 3. 更新 Scheduler (使用本轮最佳 F1)
        scheduler.step(epoch_best_f1)
        
        # 4. 保存全局最佳模型
        if epoch_best_f1 > global_best_f1:
            global_best_f1 = epoch_best_f1
            torch.save(model.state_dict(), 'best_resnet_glass.pth')
            print(f"   --> 👑 发现新高分，模型已保存 (F1: {global_best_f1:.4f})")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support() 
    train_model()