import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models 
from torch.utils.data import DataLoader
import os
import time
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import numpy as np
import seaborn as sns 

# --- Matplotlib 中文字体设置 ---
import matplotlib.font_manager as fm

desired_font_path = 'C:/Windows/Fonts/simhei.ttf' # 请根据你的操作系统和已安装字体修改此路径

font_successfully_set = False

if desired_font_path and os.path.exists(desired_font_path):
    try:
        prop = fm.FontProperties(fname=desired_font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Matplotlib: 已手动设置中文显示字体为 '{prop.get_name()}' ({desired_font_path})")
        font_successfully_set = True
    except Exception as e:
        print(f"Matplotlib: 警告：手动指定字体 '{desired_font_path}' 加载失败：{e}。将尝试自动查找。")
elif desired_font_path:
    print(f"Matplotlib: 警告：手动指定字体路径 '{desired_font_path}' 不存在。将尝试自动查找。")
else:
    print("Matplotlib: 未手动指定字体路径，将尝试自动查找。")

if not font_successfully_set:
    chinese_font_keywords = ['simhei', 'msyh', 'pingfang', 'songti', 'kaiti', 'heiti', 'cjk', 'yahei']
    auto_found_font_path = None
    
    for font_file in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        font_name = os.path.basename(font_file).lower()
        if any(keyword in font_name for keyword in chinese_font_keywords):
            try:
                prop = fm.FontProperties(fname=font_file)
                if 'chinese' in prop.get_name().lower() or 'cjk' in prop.get_name().lower() or \
                   any(keyword in prop.get_name().lower() for keyword in chinese_font_keywords):
                    auto_found_font_path = font_file
                    break
            except Exception:
                continue

    if auto_found_font_path:
        try:
            prop = fm.FontProperties(fname=auto_found_font_path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Matplotlib: 已自动设置中文显示字体为 '{prop.get_name()}' ({auto_found_font_path})")
            font_successfully_set = True
        except Exception as e:
            print(f"Matplotlib: 警告：自动找到的字体 '{auto_found_font_path}' 加载失败：{e}。将回退到默认字体。")
        
if not font_successfully_set:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    print("Matplotlib: 未找到合适的中文显示字体。中文可能显示为方块。请考虑手动指定 `desired_font_path`。")
# --- Matplotlib 中文字体设置结束 ---


# --- 全局变量和初始化 ---
loss_fig, ax_loss = plt.subplots(figsize=(6, 5))
acc_fig, ax_acc = plt.subplots(figsize=(6, 5))

roc_fig_global, ax_roc_global = plt.subplots(figsize=(6, 5))
cm_fig_global, ax_cm_global = plt.subplots(figsize=(7, 6)) # 新增混淆矩阵图

ax_roc_global.text(0.5, 0.5, "ROC曲线待生成或训练中...", horizontalalignment='center', verticalalignment='center', transform=ax_roc_global.transAxes, fontsize=12)
roc_fig_global.tight_layout()

ax_cm_global.text(0.5, 0.5, "混淆矩阵待生成或训练中...", horizontalalignment='center', verticalalignment='center', transform=ax_cm_global.transAxes, fontsize=12)
cm_fig_global.tight_layout()

train_losses_hist = []
train_accs_hist = []
val_losses_hist = []
val_accs_hist = []


# 数据加载与预处理
def load_data(data_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 增加色彩抖动
            transforms.RandomRotation(15), # 增加随机旋转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    for x in ['train', 'validation', 'test']:
        full_path = os.path.join(data_dir, x)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"缺失数据目录: {full_path}")
        if not os.listdir(full_path):
            raise ValueError(f"数据目录为空: {full_path}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'validation', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True if x == 'train' else False, num_workers=0)
                   for x in ['train', 'validation', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes

    # 计算类别权重
    class_counts = {}
    for idx in image_datasets['train'].targets:
        label = image_datasets['train'].classes[idx]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = sum(class_counts.values())
    if len(class_counts) > 0:
        class_weights = torch.tensor([total_samples / class_counts[c] for c in class_names], dtype=torch.float)
        class_weights = class_weights / class_weights.sum() * len(class_names) # 归一化，使其和为类别数
        print(f"计算得到的类别权重: {class_weights}")
    else:
        class_weights = None
        print("无法计算类别权重，因为训练集为空或没有类别。")

    return dataloaders, dataset_sizes, class_names, class_weights

# --- 新增：残差块定义 ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 残差连接
        out = self.relu(out)
        return out

# --- 改进的自定义CNN模型 (支持动态深度和宽度，可选残差连接) ---
class FlexibleCustomCNN(nn.Module):
    def __init__(self, num_classes, cnn_depth=5, base_channels=16, dropout_rate=0.5, fc_units=1024, use_residuals=False):
        super(FlexibleCustomCNN, self).__init__()
        
        self.use_residuals = use_residuals
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112
        )
        
        # 构建特征提取层
        layers = []
        in_channels = base_channels
        for i in range(cnn_depth - 1): # 减去一个，因为第一个卷积块已经定义
            out_channels = in_channels * 2
            stride = 1 # 卷积层内部不进行下采样
            if use_residuals:
                layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 每次残差块后增加一个池化
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        
        # 计算最终特征图尺寸 (224 -> 112 -> 56 -> 28 -> 14 -> 7, for 5 blocks)
        # 初始池化1次，后续(cnn_depth-1)次池化。总共 cnn_depth 次池化。
        # 224 / (2^cnn_depth)
        # 例如 depth=5: 224 / 2^5 = 224 / 32 = 7
        # 例如 depth=4: 224 / 2^4 = 224 / 16 = 14
        final_spatial_dim = 224 // (2**cnn_depth)
        
        # 检查 final_spatial_dim 是否有效，避免出现0或过小导致错误
        if final_spatial_dim == 0:
            # 当 cnn_depth 过大时，224 // (2**cnn_depth) 可能会是0
            # 这种情况通常是设计错误，或者需要AdaptiveAvgPool2d到1x1而不是0x0
            # 这里选择将其固定为1，但需要确保特征数计算正确
            print(f"警告: CNN深度 ({cnn_depth}) 过大，导致空间维度为0。将其强制设置为1。")
            final_spatial_dim = 1
            # 此时 final_channels 的计算可能不准确，因为它是基于2的幂次翻倍的
            # 但既然尺寸已强制为1x1，通常会直接使用AdaptiveAvgPool2d((1,1))
            # 并且线性层输入特征数应该基于实际的features输出
            # 这是一个更复杂的动态模型调整问题，目前简化处理
        
        self.avgpool = nn.AdaptiveAvgPool2d((final_spatial_dim, final_spatial_dim))
        
        # 计算全连接层输入特征数
        final_channels = base_channels * (2**(cnn_depth-1))
        
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * final_spatial_dim * final_spatial_dim, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, fc_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 训练模型 (这里是生成器函数，会 yield 实时更新)
def train_model_gradio(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, 
                       training_flag_gr_state_obj, 
                       train_losses_hist, train_accs_hist, val_losses_hist, val_accs_hist,
                       loss_plot_ax, acc_plot_ax, roc_plot_ax_global, roc_fig_global, cm_plot_ax_global, cm_fig_global):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses_hist.clear()
    train_accs_hist.clear()
    val_losses_hist.clear()
    val_accs_hist.clear()

    roc_plot_ax_global.clear()
    roc_plot_ax_global.text(0.5, 0.5, "训练进行中，ROC曲线待生成...", horizontalalignment='center', verticalalignment='center', transform=ax_roc_global.transAxes, fontsize=12)
    roc_fig_global.tight_layout()

    cm_plot_ax_global.clear()
    cm_plot_ax_global.text(0.5, 0.5, "训练进行中，混淆矩阵待生成...", horizontalalignment='center', verticalalignment='center', transform=ax_cm_global.transAxes, fontsize=12)
    cm_fig_global.tight_layout()

    for epoch in range(num_epochs):
        if not training_flag_gr_state_obj.value: 
            print("训练已中断!")
            # 这里的yield是给train_wrapper的for循环消费的，需要和for循环的解包数量匹配 (6个)
            yield (loss_fig.figure, acc_fig.figure, "训练已中断!", roc_fig_global, cm_fig_global, str(model))
            break 

        status_message = f'Epoch {epoch+1}/{num_epochs}'
        print(status_message)
        print('-' * 10)
        
        for phase in ['train', 'validation']:
            if not training_flag_gr_state_obj.value:
                print("训练已中断!")
                # 这里的yield也是给train_wrapper的for循环消费的，需要和for循环的解包数量匹配 (6个)
                yield (loss_fig.figure, acc_fig.figure, "训练已中断!", roc_fig_global, cm_fig_global, str(model))
                break 

            model_start_time = time.time() # 记录每个phase开始时间
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                if not training_flag_gr_state_obj.value:
                    print("训练已中断!")
                    # 这里的yield也是给train_wrapper的for循环消费的，需要和for循环的解包数量匹配 (6个)
                    yield (loss_fig.figure, acc_fig.figure, "训练已中断!", roc_fig_global, cm_fig_global, str(model))
                    break 

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if not training_flag_gr_state_obj.value: 
                break

            if phase == 'train' and scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(running_loss / len(dataloaders[phase].dataset)) # ReduceLROnPlateau 需要监控指标
                else:
                    scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - model_start_time
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} 耗时: {time_elapsed:.0f}秒')
            
            if phase == 'train':
                train_losses_hist.append(epoch_loss)
                train_accs_hist.append(float(epoch_acc))
            else: # val
                val_losses_hist.append(epoch_loss)
                val_accs_hist.append(float(epoch_acc))
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
        if not training_flag_gr_state_obj.value: 
            break

        # 实时更新图表
        loss_plot_ax.clear()
        loss_plot_ax.plot(train_losses_hist, label='训练损失')
        loss_plot_ax.plot(val_losses_hist, label='验证损失')
        loss_plot_ax.set_title('损失曲线')
        loss_plot_ax.set_xlabel('Epoch')
        loss_plot_ax.set_ylabel('Loss')
        loss_plot_ax.legend()
        loss_fig.tight_layout()

        acc_plot_ax.clear()
        acc_plot_ax.plot(train_accs_hist, label='训练准确率')
        acc_plot_ax.plot(val_accs_hist, label='验证准确率')
        acc_plot_ax.set_title('准确率曲线')
        acc_plot_ax.set_xlabel('Epoch')
        acc_plot_ax.set_ylabel('Accuracy')
        acc_plot_ax.legend()
        acc_fig.tight_layout()
        
        status_message = (f"训练中... 轮次: {epoch+1}/{num_epochs}\n"
                          f"训练损失: {train_losses_hist[-1]:.4f} 训练准确率: {train_accs_hist[-1]:.4f}\n"
                          f"验证损失: {val_losses_hist[-1]:.4f} 验证准确率: {val_accs_hist[-1]:.4f}")
        
        # 这里的yield也是给train_wrapper的for循环消费的，需要和for循环的解包数量匹配 (6个)
        yield loss_fig.figure, acc_fig.figure, status_message, roc_fig_global, cm_fig_global, str(model)

    # 训练结束后加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print("训练完成!")
    return model

# 评估模型 (保持不变，因为它已经很完善)
def evaluate_model(model, dataloader, device, class_names, roc_ax_to_plot, roc_fig_to_plot, cm_ax_to_plot, cm_fig_to_plot):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. 计算核心指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 2. 生成分类报告 (包含每个类别的指标)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    # 3. 绘制混淆矩阵
    cm_ax_to_plot.clear()
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=cm_ax_to_plot)
    cm_ax_to_plot.set_xlabel('预测标签')
    cm_ax_to_plot.set_ylabel('真实标签')
    cm_ax_to_plot.set_title('混淆矩阵')
    cm_fig_to_plot.tight_layout()
    print("混淆矩阵成功绘制。")

    # 4. 绘制ROC曲线 (二分类或One-vs-Rest多分类)
    roc_ax_to_plot.clear()
    roc_auc = None
    
    if len(class_names) == 2:
        if all_probs.ndim == 2 and all_probs.shape[1] == 2:
            try:
                if len(np.unique(all_labels)) == 2:
                    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
                    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                    roc_ax_to_plot.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
                    roc_ax_to_plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    roc_ax_to_plot.set_xlim([0.0, 1.0])
                    roc_ax_to_plot.set_ylim([0.0, 1.05])
                    roc_ax_to_plot.set_xlabel('假阳性率')
                    roc_ax_to_plot.set_ylabel('真阳性率')
                    roc_ax_to_plot.set_title('接收者操作特征曲线')
                    roc_ax_to_plot.legend(loc="lower right")
                    print(f"ROC曲线成功绘制，AUC: {roc_auc:.4f}")
                else:
                    roc_ax_to_plot.text(0.5, 0.5, "ROC曲线绘制错误: 测试集只包含单个类别", horizontalalignment='center', verticalalignment='center', transform=roc_ax_to_plot.transAxes, fontsize=10)
                    print(f"ROC曲线未绘制: 测试集只包含单个类别。Unique labels: {np.unique(all_labels)}")
            except ValueError as e:
                roc_ax_to_plot.text(0.5, 0.5, f"ROC曲线绘制错误: {e}", horizontalalignment='center', verticalalignment='center', transform=roc_ax_to_plot.transAxes, fontsize=10)
                print(f"ROC曲线绘制错误: {e}")
        else:
            roc_ax_to_plot.text(0.5, 0.5, "无法为非二分类或不完整概率数据绘制ROC曲线", horizontalalignment='center', verticalalignment='center', transform=roc_ax_to_plot.transAxes, fontsize=10)
            print(f"ROC曲线未绘制: 预测概率数组形状不正确 (预期 N,2)，实际 {all_probs.shape}")
    elif len(class_names) > 2: # 多分类One-vs-Rest ROC
        roc_ax_to_plot.set_title('接收者操作特征曲线 (One-vs-Rest)')
        roc_ax_to_plot.set_xlabel('假阳性率')
        roc_ax_to_plot.set_ylabel('真阳性率')
        roc_ax_to_plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        all_aucs = []
        for i, class_name in enumerate(class_names):
            if i < all_probs.shape[1]: 
                try:
                    binary_labels = (all_labels == i).astype(int)
                    if len(np.unique(binary_labels)) == 2:
                        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
                        class_auc = roc_auc_score(binary_labels, all_probs[:, i])
                        roc_ax_to_plot.plot(fpr, tpr, lw=1, label=f'类别 {class_name} (AUC = {class_auc:.2f})')
                        all_aucs.append(class_auc)
                    else:
                        print(f"类别 '{class_name}' 在测试集中仅包含一种标签，无法计算ROC曲线。")
                except ValueError as e:
                    print(f"为类别 '{class_name}' 绘制ROC曲线时出错: {e}")
            else:
                print(f"类别索引 {i} 超出概率数组范围 {all_probs.shape[1]}")

        roc_ax_to_plot.legend(loc="lower right", fontsize='small')
        roc_auc = np.mean(all_aucs) if all_aucs else None
        print(f"多分类ROC曲线绘制完成，平均AUC: {roc_auc:.4f}")

    else:
        roc_ax_to_plot.text(0.5, 0.5, "ROC曲线需要至少两个类别", horizontalalignment='center', verticalalignment='center', transform=roc_ax_to_plot.transAxes, fontsize=12)
        print("ROC曲线未绘制: 类别数量不足。")

    roc_fig_to_plot.tight_layout()

    return {
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'class_report': class_report,
        'roc_curve_fig': roc_fig_to_plot,
        'roc_auc': roc_auc,
        'confusion_matrix_fig': cm_fig_to_plot,
    }

def create_ui():
    with gr.Blocks(title="中药图像分类系统") as demo:
        training_flag = gr.State(False)
        current_model = gr.State(None)
        class_names_state = gr.State([])
        class_weights_state = gr.State(None)

        def predict_image(image_path, model_value, class_names_list):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            if image_path is None:
                return "请上传图片"
            if model_value is None:
                return "模型未训练或未加载"
            if not class_names_list:
                return "类别信息未加载，请先训练模型"

            try:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img = Image.open(image_path).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)

                model_value.eval()
                with torch.no_grad():
                    outputs = model_value(img)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    _, preds = torch.max(outputs, 1)
                    predicted_class_idx = preds[0].item()
                    
                    predicted_class = class_names_list[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx].item() * 100

                all_confidences = {class_names_list[i]: f"{probabilities[i].item()*100:.2f}%" for i in range(len(class_names_list))}
                result_str = f"预测结果: {predicted_class} (置信度: {confidence:.2f}%)\n\n所有类别置信度:\n"
                for cls_name, conf in all_confidences.items():
                    result_str += f"{cls_name}: {conf}\n"

                return result_str
            except Exception as e:
                return f"预测错误: {str(e)}"

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 模型训练控制")

                with gr.Accordion("数据集设置", open=True):
                    data_dir = gr.Textbox(
                        label="数据集路径", 
                        value=os.path.join(os.path.expanduser('~'), 'Desktop', '大作业', '中药数据集'),
                        interactive=True
                    )

                    dataset_info = gr.Textbox(label="数据集信息", interactive=False)

                    def update_dataset_info(data_path):
                        try:
                            train_path = os.path.join(data_path, "train")
                            if os.path.exists(train_path) and os.listdir(train_path):
                                temp_dataset = datasets.ImageFolder(train_path)
                                class_names_temp = temp_dataset.classes
                                return f"数据集有效\n类别数: {len(class_names_temp)}\n类别列表: {', '.join(class_names_temp)}"
                            return "无效的数据集路径或train目录为空"
                        except Exception as e:
                            return f"读取数据集出错: {e}"

                    data_dir.change(
                        fn=update_dataset_info,
                        inputs=data_dir,
                        outputs=dataset_info
                    )

                with gr.Accordion("训练参数设置", open=True):
                    with gr.Row():
                        batch_size = gr.Slider(8, 128, value=32, step=8, label="批大小")
                        num_epochs = gr.Slider(1, 100, value=10, step=1, label="训练轮数")
                    with gr.Row():
                        learning_rate = gr.Number(value=0.001, label="学习率")
                        dropout_rate = gr.Slider(0.1, 0.8, value=0.5, step=0.1, label="Dropout率")
                    with gr.Row():
                        weight_decay = gr.Number(value=1e-4, label="权重衰减 (L2正则化)")
                        use_class_weights = gr.Checkbox(value=False, label="使用类别权重平衡", info="处理类别不平衡")
                    with gr.Row():
                        scheduler_type = gr.Dropdown(
                            ["None", "StepLR", "ReduceLROnPlateau"], 
                            value="ReduceLROnPlateau", label="学习率调度器" # 默认改为 ReduceLROnPlateau
                        )

                with gr.Accordion("模型架构设置 (自定义CNN)", open=True): 
                    with gr.Row():
                        cnn_depth = gr.Slider(2, 8, value=5, step=1, label="CNN深度 (卷积块数量)", info="每个卷积块包含一个池化层")
                        base_channels = gr.Slider(8, 64, value=16, step=8, label="初始通道数", info="后续通道数会翻倍")
                    with gr.Row():
                        fc_units = gr.Slider(256, 2048, value=1024, step=256, label="全连接单元数")
                        use_residuals = gr.Checkbox(value=False, label="使用残差连接", info="对深层网络更稳定") # 新增

                train_btn = gr.Button("开始训练", variant="primary")
                stop_btn = gr.Button("停止训练")

            with gr.Column():
                gr.Markdown("## 训练与评估结果")

                with gr.Tabs():
                    with gr.Tab("训练过程"):
                        with gr.Row():
                            loss_plot = gr.Plot(label="损失曲线", value=loss_fig)
                            acc_plot = gr.Plot(label="准确率曲线", value=acc_fig)

                    with gr.Tab("评估指标"):
                        metrics = gr.Textbox(label="评估指标概览", interactive=False)
                        detailed_metrics = gr.Textbox(label="详细分类报告 (每类指标)", interactive=False, lines=10)
                        roc_plot = gr.Plot(label="ROC曲线", value=roc_fig_global)
                        cm_plot = gr.Plot(label="混淆矩阵", value=cm_fig_global)

                    with gr.Tab("模型信息"):
                        model_summary = gr.Textbox(label="模型结构", interactive=False)

                with gr.Accordion("测试样本预测", open=True):
                    with gr.Row():
                        test_image = gr.Image(label="上传测试图片", type="filepath")
                        test_result = gr.Label(label="预测结果")


        def train_wrapper(data_dir_path, batch_size, num_epochs, learning_rate, dropout_rate, 
                          weight_decay, use_class_weights, scheduler_type, 
                          cnn_depth, base_channels, fc_units, use_residuals):
            
            if training_flag.value:
                # 训练已在进行，返回当前UI内容，避免重复启动
                # 确保这里返回8个值，与outputs参数匹配
                yield ax_loss.figure, ax_acc.figure, "训练已在进行中...", "", roc_fig_global, cm_fig_global, "模型结构: 训练中...", True 
                return
            
            training_flag.value = True 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            try:
                # 1. 加载数据并计算类别权重
                dataloaders, _, class_names, calculated_class_weights = load_data(data_dir_path, batch_size)
                class_names_state.value = class_names 
                if use_class_weights and calculated_class_weights is not None:
                    class_weights_state.value = calculated_class_weights.to(device)
                else:
                    class_weights_state.value = None
                
                # 2. 创建模型 (使用新的 FlexibleCustomCNN)
                num_classes = len(class_names)
                model = FlexibleCustomCNN(num_classes, cnn_depth=cnn_depth, base_channels=base_channels, 
                                          dropout_rate=dropout_rate, fc_units=fc_units, 
                                          use_residuals=use_residuals).to(device)
                
                # 3. 定义损失函数、优化器和调度器
                criterion = nn.CrossEntropyLoss(weight=class_weights_state.value) 
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                scheduler = None
                if scheduler_type == "StepLR":
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                elif scheduler_type == "ReduceLROnPlateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

                # 初始 yield，通知 UI 训练开始，ROC/CM图显示全局占位符
                # 确保这里返回8个值，与outputs参数匹配
                yield loss_fig, acc_fig, "训练准备中...", "", roc_fig_global, cm_fig_global, str(model), True
                
                # 4. 训练模型并实时更新图表
                for loss_fig_val, acc_fig_val, status_msg_val, roc_plot_val, cm_plot_val, model_summary_val in \
                    train_model_gradio( # train_model_gradio 内部每次 yield 6个值
                        model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, 
                        training_flag, 
                        train_losses_hist, train_accs_hist, val_losses_hist, val_accs_hist,
                        ax_loss, ax_acc, ax_roc_global, roc_fig_global, ax_cm_global, cm_fig_global
                    ):
                    # train_wrapper 接收 train_model_gradio 的 6 个值，并补齐为 8 个值 yield 给 Gradio
                    yield loss_fig_val, acc_fig_val, status_msg_val, "", roc_plot_val, cm_plot_val, model_summary_val, training_flag.value
                    
                    if not training_flag.value:
                        print("训练被用户中断，跳过最终评估。")
                        # 确保这里返回8个值，与outputs参数匹配
                        yield loss_fig.figure, acc_fig.figure, "训练已中断!", "", roc_fig_global, cm_fig_global, "模型结构: 训练中断", False
                        return

                # 5. 训练结束后，模型已经加载了最佳权重
                # 6. 评估模型 (使用测试数据)
                eval_results = evaluate_model(model, dataloaders['test'], device, class_names, 
                                              ax_roc_global, roc_fig_global, ax_cm_global, cm_fig_global)
                
                metrics_output = f"""训练完成!
                                准确率: {eval_results['accuracy']:.4f}
                                加权平均精确率: {eval_results['precision_weighted']:.4f}
                                加权平均召回率: {eval_results['recall_weighted']:.4f}
                                加权平均F1分数: {eval_results['f1_weighted']:.4f}
                                """
                if eval_results['roc_auc'] is not None:
                    metrics_output += f"\n平均AUC: {eval_results['roc_auc']:.4f}"
                else:
                    metrics_output += f"\nAUC: N/A (非二分类或数据不足)"
                
                detailed_metrics_output = ""
                if 'class_report' in eval_results:
                    report_str = ""
                    for class_name, metrics_dict in eval_results['class_report'].items():
                        if isinstance(metrics_dict, dict):
                            report_str += f"类别: {class_name}\n"
                            for metric_name, value in metrics_dict.items():
                                if isinstance(value, float):
                                    report_str += f"  {metric_name}: {value:.4f}\n"
                                else:
                                    report_str += f"  {metric_name}: {value}\n"
                        elif class_name in ['accuracy', 'macro avg', 'weighted avg']: 
                             if isinstance(metrics_dict, float):
                                report_str += f"{class_name}: {metrics_dict:.4f}\n"
                             else: 
                                report_str += f"{class_name}:\n"
                                for k, v in metrics_dict.items():
                                     if isinstance(v, float):
                                         report_str += f"  {k}: {v:.4f}\n"
                                     else:
                                         report_str += f"  {k}: {v}\n"
                    detailed_metrics_output = report_str

                # 最终的 yield，包含评估结果和绘制好的图表
                # 确保这里返回8个值，与outputs参数匹配
                yield loss_fig, acc_fig, metrics_output, detailed_metrics_output, eval_results['roc_curve_fig'], eval_results['confusion_matrix_fig'], str(model), False

                current_model.value = model # 存储训练好的模型供预测使用

            except FileNotFoundError as e:
                err_msg = f"错误: 数据集路径不完整或不存在。{str(e)}"
                print(err_msg)
                ax_roc_global.clear()
                ax_roc_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_roc_global.transAxes, fontsize=10, color='red')
                roc_fig_global.tight_layout()
                ax_cm_global.clear()
                ax_cm_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_cm_global.transAxes, fontsize=10, color='red')
                cm_fig_global.tight_layout()
                # 确保这里返回8个值，与outputs参数匹配
                yield None, None, err_msg, "", roc_fig_global, cm_fig_global, "模型结构显示错误", False
            except ValueError as e:
                err_msg = f"错误: 数据集内容无效。{str(e)}"
                print(err_msg)
                ax_roc_global.clear()
                ax_roc_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_roc_global.transAxes, fontsize=10, color='red')
                roc_fig_global.tight_layout()
                ax_cm_global.clear()
                ax_cm_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_cm_global.transAxes, fontsize=10, color='red')
                cm_fig_global.tight_layout()
                # 确保这里返回8个值，与outputs参数匹配
                yield None, None, err_msg, "", roc_fig_global, cm_fig_global, "模型结构显示错误", False
            except Exception as e:
                err_msg = f"训练或评估出错: {str(e)}（请检查数据集路径和参数设置）"
                print(err_msg)
                ax_roc_global.clear()
                ax_roc_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_roc_global.transAxes, fontsize=10, color='red')
                roc_fig_global.tight_layout()
                ax_cm_global.clear()
                ax_cm_global.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax_cm_global.transAxes, fontsize=10, color='red')
                cm_fig_global.tight_layout()
                # 确保这里返回8个值，与outputs参数匹配
                yield None, None, err_msg, "", roc_fig_global, cm_fig_global, "模型结构显示错误", False
            finally:
                training_flag.value = False 

            
        train_btn.click(
            fn=train_wrapper,
            inputs=[data_dir, batch_size, num_epochs, learning_rate, dropout_rate, 
                    weight_decay, use_class_weights, scheduler_type, 
                    cnn_depth, base_channels, fc_units, use_residuals],
            outputs=[loss_plot, acc_plot, metrics, detailed_metrics, roc_plot, cm_plot, model_summary, training_flag], 
            queue=True, 
            api_name="start_training"
        )

        def stop_training():
            training_flag.value = False
            return False 

        stop_btn.click(
            fn=stop_training,
            inputs=[],
            outputs=[training_flag], # 这里只更新 training_flag 状态
            queue=False
        )

        test_image.change(
            fn=predict_image,
            inputs=[test_image, current_model, class_names_state], 
            outputs=test_result
        )

    return demo

if __name__ == "__main__":
    data_dir_path_check = os.path.join(os.path.expanduser('~'), 'Desktop', '大作业', '中药数据集')
    if not os.path.exists(data_dir_path_check):
        print(f"警告: 数据集路径 '{data_dir_path_check}' 不存在。请确保您的中药数据集位于桌面上的 '中药数据集' 文件夹中，且包含 train, val, test 子目录。")
        print("Gradio 应用将启动，但训练功能可能无法正常工作。")

    ui = create_ui()
    ui.launch()
