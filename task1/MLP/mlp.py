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
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# hello world
# =================配置参数=================
CONFIG = {
    'input_size': 256,      # 调整尺寸以适应 MLP 和 4GB 显存
    'batch_size': 128,       # 较小的 batch size 适应显存
    'lr': 0.005,
    'epochs': 20,
    'device': 'cuda' ,
    'train_dir': 'D:\\table\ml\project\dataset_try\\train', 
    'test_dir': 'D:\\table\ml\project\dataset_try\\test' ,  
    #'train_dir': 'D:\\table\机器学习\project\dataset\\train', 
    #'test_dir': 'D:\\table\机器学习\project\dataset\\test' ,  
}

print(f"Using device: {CONFIG['device']}")

# =================数据处理=================
class GlassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        根据文档:
        dataset/train/img/ 包含图片
        dataset/train/txt/ 包含标签
        """
        self.img_dir = os.path.join(root_dir, 'img')
        self.txt_dir = os.path.join(root_dir, 'txt')
        self.transform = transform
        
        # 获取所有图片路径
        self.img_paths = glob.glob(os.path.join(self.img_dir, '*.png')) # [cite: 16]
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path)
        txt_name = file_name.replace('.png', '.txt')
        txt_path = os.path.join(self.txt_dir, txt_name)
        
        # 标签逻辑: 如果txt文件存在则为1(Defective), 否则为0(Non-defective) 
        label = 1.0 if os.path.exists(txt_path) else 0.0
        
        image = Image.open(img_path).convert('L') # 转为灰度图 (L mode) 减少维度
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# 数据增强与预处理
# 将图像Flatten拉平以适应MLP输入
transform = transforms.Compose([
    transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
    transforms.ToTensor(),
    # 归一化有助于 MLP 收敛
    transforms.Normalize(mean=[0.5], std=[0.5]), 
    transforms.Lambda(lambda x: torch.flatten(x)) # [1, 128, 128] -> [16384]
])

# =================模型定义=================
class GlassMLP(nn.Module):
    def __init__(self, input_dim):
        super(GlassMLP, self).__init__()
        # 这种深层宽结构有助于拟合，BN和Dropout防止过拟合
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1) # 输出层，配合 BCEWithLogitsLoss 不需要 Sigmoid
        )
        
    def forward(self, x):
        return self.net(x)

# =================训练与评估=================
def calculate_metrics(y_true, y_pred):
    # 将概率转为 0/1 预测
    y_pred_tag = np.round(y_pred)
    
    # 计算 Precision, Recall, F1 (针对 Defective 类) [cite: 44-47]
    p = precision_score(y_true, y_pred_tag, zero_division=0)
    r = recall_score(y_true, y_pred_tag, zero_division=0)
    f1 = f1_score(y_true, y_pred_tag, zero_division=0)
    return p, r, f1


def find_best_threshold(y_true, y_probs):
    best_f1 = 0
    best_thresh = 0.5
    
    # 遍历从 0.1 到 0.9 的所有阈值
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1

def train_and_evaluate():
    set_seed(42)
    # 1. 准备数据
    train_dataset = GlassDataset(CONFIG['train_dir'], transform=transform)
    test_dataset = GlassDataset(CONFIG['test_dir'], transform=transform)
    
    # 简单的加权采样或计算 pos_weight 来处理不平衡
    # 这里我们计算正负样本比例传给 Loss
    num_pos = sum([1 for _, label in train_dataset if label == 1])
    num_neg = len(train_dataset) - num_pos
    pos_weight = torch.tensor([2.0*num_neg / num_pos]).to(CONFIG['device'])
    print(f"Dataset stats: Positive: {num_pos}, Negative: {num_neg}, Pos Weight: {pos_weight.item():.2f}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # 2. 初始化模型
    input_dim = CONFIG['input_size'] * CONFIG['input_size'] # 128*128 = 16384
    model = GlassMLP(input_dim).to(CONFIG['device'])
    
    # 3. 损失函数与优化器
    # 使用带权重的 BCE Loss 来解决类别不平衡 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 4. 训练循环
    best_f1 = 0.0
    start_time = time.time()
    for epoch in range(CONFIG['epochs']):
        print(f"training Epoch {epoch+1}/{CONFIG['epochs']}...")
        model.train()
        train_loss = 0
        test_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            labels = labels.unsqueeze(1) # [batch] -> [batch, 1]
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 5. 每个 Epoch 进行评估
        if (epoch + 1) % 1 == 0:
            model.eval()
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(CONFIG['device'])
                    outputs = model(imgs)
                    probs = torch.sigmoid(outputs) # Logits -> Probabilities
                    
                    y_true.extend(labels.numpy())
                    y_pred.extend(probs.cpu().numpy())
                    loss = criterion(outputs, labels.unsqueeze(1).to(CONFIG['device']))
                    test_loss += loss.item()
            
            # 计算指标
            p, r, f1 = calculate_metrics(y_true, y_pred)
            
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | train Loss: {train_loss/len(train_loader):.4f} | test loss:{test_loss/len(test_loader):.4f} | "
                f"Test F1: {f1:.4f} (P: {p:.4f}, R: {r:.4f})")
            #best_t, best_f1 = find_best_threshold(y_true, y_pred)
            #print(f"最佳阈值: {best_t}, 对应的最高 F1: {best_f1}")
            if f1 > best_f1:
                best_f1 = f1
                # 保存模型，为后续手写 Task 1 做参考
                torch.save(model.state_dict(), 'best_mlp_model.pth') 

    print(f"\nTraining Finished in {(time.time()-start_time):.2f}s. Best Test F1: {best_f1:.4f}")

if __name__ == '__main__':
    # 确保路径存在，避免报错
    if not os.path.exists(CONFIG['train_dir']):
        print(f"Error: Dataset not found at {CONFIG['train_dir']}. Please download dataset first.")
    else:
        train_and_evaluate()