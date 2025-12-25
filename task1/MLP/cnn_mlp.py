import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time
import tqdm

# ================= 配置参数 (针对 4GB 显存优化) =================
CONFIG = {
    'img_size': 256,        # 保持 256，通过 GAP 和 AMP 节省显存
    'batch_size': 64,       # 优化后显存占用降低，可以尝试开大 BatchSize 加速
    'lr': 0.001,
    'epochs': 40,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_dir': r'D:\table\ml\project\dataset\train', 
    'test_dir': r'D:\table\ml\project\dataset\test',
    'pos_weight': 9.0,
    'num_workers': 4        # 【优化】开启多进程读取
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
        
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ================= 模型架构：轻量化优化版 =================
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        
        # 通道数减半策略 (32->16, 64->32...) 以适应 4GB 显存并加速
        # 如果觉得精度不够，可以把通道数改回去 (16->32, 32->64...)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # 【优化关键】全局平均池化 (Global Average Pooling)
        # 无论前面特征图多大，这里都变成 [batch, 256, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), # 参数量极小：256*64，不再是 32768*1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x) # 压缩特征
        x = self.classifier(x)
        return x

# ================= 训练与评估函数 (集成 AMP) =================
def train_model():
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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
    
    # 【优化】开启 pin_memory 和 num_workers
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = OptimizedCNN().to(CONFIG['device'])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG['pos_weight']]).to(CONFIG['device']))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 【优化】混合精度 Scaler
    scaler = torch.cuda.amp.GradScaler()

    print(f"开始训练，设备: {CONFIG['device']} | AMP: On | Num_workers: {CONFIG['num_workers']}")
    
    best_f1 = 0
    start_time = time.time()

    for epoch in range(CONFIG['epochs']):
        #print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        model.train()
        total_loss = 0
        
        for imgs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False):
            imgs = imgs.to(CONFIG['device'], non_blocking=True) # non_blocking 加速数据传输
            labels = labels.to(CONFIG['device'], non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # 【优化】混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # 【优化】混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        # 验证 (也可以用 AMP 加速推理)
        model.eval()
        all_labels = []
        all_preds = []
        test_loss = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(CONFIG['device'], non_blocking=True)
                labels = labels.to(CONFIG['device'], non_blocking=True).unsqueeze(1)
                
                with torch.cuda.amp.autocast(): # 推理也开 AMP
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                all_labels.extend(labels.cpu().numpy()) # 记得移回 CPU
                all_preds.extend(preds)
                test_loss += loss.item()

        # 计算指标
        # 注意：这里把 list 转 numpy 再算，防止报错
        all_labels = np.array(all_labels).flatten()
        all_preds = np.array(all_preds).flatten()
        
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        p = precision_score(all_labels, all_preds, zero_division=0)
        r = recall_score(all_labels, all_preds, zero_division=0)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] ({epoch_time:.0f}s) "
              f"Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {test_loss/len(test_loader):.4f} | "
              f"F1: {f1:.4f} (P: {p:.4f}, R: {r:.4f})")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_optimized_cnn.pth')
            print("--> 👑 保持最佳模型")

if __name__ == '__main__':
    # 解决 Windows 下 num_workers 报错的关键
    torch.multiprocessing.freeze_support() 
    train_model()