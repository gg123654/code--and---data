"""
训练脚本：使用 AMSDAL 损失函数训练 NEU-DET 数据集

AMSDAL (Adaptive Multi-Scale Defect-Aware Loss) 是专门为钢材表面缺陷检测设计的创新损失函数。

使用方法:
    python train_with_amsdal.py
"""

from ultralytics import YOLO
from ultralytics.utils.neu_det_loss import AMSDALoss


def train_with_amsdal():
    """使用 AMSDAL 损失函数训练模型"""
    
    # 加载模型
    print("正在加载模型...")
    model = YOLO('yolo11n.pt')  # 或使用其他预训练模型: yolo11s.pt, yolo11m.pt 等
    
    # 确保模型已去并行化（de-paralleled）
    if hasattr(model.model, 'module'):
        model.model = model.model.module
    
    # 替换损失函数为 AMSDAL
    print("正在初始化 AMSDAL 损失函数...")
    model.model.loss = AMSDALoss(
        model.model,
        tal_topk=10,                    # Task Aligned Assigner 的 top-k 值
        small_target_threshold=0.05,     # 小目标阈值（相对于图像大小）
        scale_weight_power=1.5,         # 尺度加权幂次，值越大对小目标关注越多
        boundary_weight=2.0,             # 边界精度损失权重
        class_adaptation_rate=0.01,      # 类别难度适应率
        consistency_weight=0.1          # 多尺度一致性损失权重
    )
    
    print("开始训练...")
    # 训练参数
    results = model.train(
        data='data.yaml',               # 数据集配置文件
        epochs=100,                     # 训练轮数
        imgsz=640,                      # 图像大小
        batch=16,                       # 批次大小（根据GPU内存调整）
        lr0=0.01,                       # 初始学习率
        device=0,                       # GPU设备ID（使用CPU则设为'cpu'）
        project='runs/train',           # 项目目录
        name='neu_det_amsdal',         # 实验名称
        save=True,                      # 保存检查点
        save_period=10,                 # 每N个epoch保存一次
        val=True,                       # 验证
        plots=True,                     # 生成训练图表
        verbose=True,                   # 详细输出
    )
    
    print("训练完成！")
    print(f"最佳模型保存在: {results.save_dir}")
    
    return results


if __name__ == '__main__':
    train_with_amsdal()

