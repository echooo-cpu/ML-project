import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler # 引入混合精度
from tqdm import tqdm # 引入进度条
import time

# ================= 极速配置 =================
DATASET_ROOT = "D:\\table\机器学习\project\dataset\dataset"  
BATCH_SIZE = 64       # 显存4G下，128x128图片通常可以开到64或128
IMAGE_SIZE = 128      # 核心加速点：从340降到128
LEARNING_RATE = 0.001
EPOCHS = 15            # 4万张图，跑5轮通常足够收敛看趋势
NUM_WORKERS = 4       # 设置为你的CPU核心数，加速读取
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据集 (保持不变) =================
class GlassDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.img_dir = os.path.join(root_dir, mode, 'img')
        self.txt_dir = os.path.join(root_dir, mode, 'txt')
        self.transform = transform
        
        # 快速扫描文件，不检查全部txt存在性以加速初始化（假设文件名对应）
        # 仅在__getitem__时判断
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        txt_path = os.path.join(self.txt_dir, img_name.replace('.png', '.txt'))
        
        # 1. 极速判定标签 (IO操作)
        label = 1.0 if os.path.exists(txt_path) else 0.0
        
        # 2. 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 容错处理，防止坏图崩代码
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# ================= 轻量化模型 (FastCNN) =================
class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # 加BN层加速收敛
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16
        )
        
        # 自适应池化：不管前面特征图多大，强行变成 4x4
        # 这样我们可以随意改输入图片大小而不用改全连接层代码
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), # 参数量极小
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# ================= 训练流程 (带AMP加速) =================
def train_fast():
    print(f"开始极速训练 | Device: {DEVICE} | Image Size: {IMAGE_SIZE}")
    
    # 1. 预处理
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # 简单的归一化
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. 加载器
    train_dataset = GlassDataset(DATASET_ROOT, mode='train', transform=transform)
    # 为了速度，我们只取Test集的前500张做验证，不跑全量Test
    test_dataset = GlassDataset(DATASET_ROOT, mode='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"训练集大小: {len(train_dataset)} | Batch Size: {BATCH_SIZE}")
    
    # 3. 初始化
    model = FastCNN().to(DEVICE)
    # 使用你提供的权重，或者根据实际情况动态计算
    pos_weight = torch.tensor([9.78]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # 混合精度Scaler

    # 4. 循环
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        losses = []
        
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast(): 
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        
        avg_loss = sum(losses) / len(losses)
        
        # 快速验证
        f1 = quick_evaluate(model, test_loader)
        print(f"Epoch {epoch+1} 完成. Avg Loss: {avg_loss:.4f} | Test F1 (Defective): {f1:.4f}")

def quick_evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # 只跑前20个Batch看个大概，节省时间
        # 如果你想看精确结果，去掉 zip(..., range(20))
        for i, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            # if i > 20: break # 取消注释这行可以只测一小部分
            
    return f1_score(all_labels, all_preds, average='binary')

if __name__ == "__main__":
    # Windows下多进程必须放在if __name__里
    if not os.path.exists(DATASET_ROOT):
        print("请设置正确的 DATASET_ROOT 路径")
    else:
        start = time.time()
        train_fast()
        print(f"总耗时: {(time.time() - start)/60:.2f} 分钟")