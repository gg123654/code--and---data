import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import pandas as pd
from tabulate import tabulate
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"#
import shutil
from datetime import datetime
# 导入增强版损失函数
from ultralytics.utils.neu_det_loss_enhanced import EnhancedAMSDALoss

if __name__ == '__main__':
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用脚本所在目录下的runs文件夹
    script_dir = os.path.dirname(__file__)
    save_dir = os.path.join(script_dir, 'runs', 'train', f'exp_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📁 训练结果将保存到: {save_dir}")
    print(f"📁 项目根目录: {os.getcwd()}")
    
    # 加载模型
    print("🔧 正在加载模型...")
    model = YOLO('new_moudle.yaml')
    
    # 应用增强版损失函数 EnhancedAMSDALoss
    print("🚀 正在应用 EnhancedAMSDALoss 损失函数...")
    try:
        # 确保模型已去并行化（de-paralleled）
        if hasattr(model.model, 'module'):
            model.model = model.model.module
        
        # 替换损失函数为 EnhancedAMSDALoss
        model.model.loss = EnhancedAMSDALoss(
            model.model,
            tal_topk=20,                    # Task Aligned Assigner 的 top-k 值：从15提升到20，大幅增加正样本匹配
            small_target_threshold=0.01,     # 小目标阈值：从0.015降低到0.01，极关注极小目标
            scale_weight_power=3.0,         # 尺度加权幂次：从2.8提升到3.0，极大幅度提升小目标/细长目标权重
            boundary_weight=3.5,             # 边界精度损失权重：从3.0提升到3.5，极强化边界精度
            class_adaptation_rate=0.08,      # 类别难度适应率：从0.05提升到0.08，极快适应难类
            consistency_weight=0.25,          # 多尺度一致性损失权重：从0.2提升到0.25
            shape_power=1.8,                 # 形状感知权重幂次：从1.6提升到1.8，极强化形状感知
            contrast_power=2.8,               # 对比度自适应权重幂次：从2.5提升到2.8，极强化低对比度缺陷
            use_shape_aware=True,            # 启用形状感知损失
            use_contrast_adaptive=True,      # 启用对比度自适应损失
            use_severity_aware=True          # 启用严重性感知损失
        )
        print("✅ EnhancedAMSDALoss 损失函数已成功应用！")
        print("   - 形状感知损失: 启用")
        print("   - 对比度自适应损失: 启用")
        print("   - 严重性感知损失: 启用")
    except Exception as e:
        print(f"⚠️  应用增强损失函数时出错: {e}")
        print("   将使用默认损失函数继续训练...")
        import traceback
        traceback.print_exc()
    # 执行训练
    results = model.train(
        data=os.path.join(os.path.dirname(__file__), 'data.yaml'),
        cache=False,
        imgsz=640,
        epochs=500,
        batch=16,
        close_mosaic=10,
        workers=0,
        device='0',
        optimizer='SGD',
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率（降低学习率衰减，保持更高学习率）
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=12.0,  # 边界框损失权重：从10.0提升到12.0，极大幅度强化定位（提升召回率）
        cls=1.0,  # 分类损失权重：从0.8提升到1.0，极强化分类（提升召回率）
        dfl=2.5,  # DFL损失权重：从2.0提升到2.5，极强化边界框回归
        hsv_h=0.015,  # 色调增强：略微提升数据增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=10.0,  # 旋转角度：略微提升
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=2.0,  # 剪切
        perspective=0.0,  # 透视变换
        flipud=0.0,  # 上下翻转
        fliplr=0.5,  # 左右翻转
        mosaic=1.0,  # Mosaic增强
        mixup=0.1,  # Mixup增强：略微提升
        copy_paste=0.1,  # Copy-paste增强
        amp=False,                                        
        patience=120,  # 延长早停耐心：从100提升到120，给难例更多训练时间
        conf=0.2,  # 验证时的置信度阈值：从0.25降低到0.2，极大幅度提升召回率
        iou=0.4,  # NMS的IoU阈值：从0.45降低到0.4，保留更多检测框
        max_det=1000,  # 最大检测数量：增加以提升召回率
        project=save_dir,
        save_period=-1,
    )
    # 保存模型
    best_model_path = os.path.join(save_dir, 'yolov8_training', 'weights', 'best.pt')
    last_model_path = os.path.join(save_dir, 'yolov8_training', 'weights', 'last.pt')
    # 创建导出目录
    export_dir = os.path.join(save_dir, 'exported_models')
    os.makedirs(export_dir, exist_ok=True)

    # 复制最佳模型和最终模型到导出目录
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, os.path.join(export_dir, 'best_model.pt'))
        print(f"最佳模型已保存至: {os.path.join(export_dir, 'best_model.pt')}")

    if os.path.exists(last_model_path):
        shutil.copy(last_model_path, os.path.join(export_dir, 'last_model.pt'))
        print(f"最终模型已保存至: {os.path.join(export_dir, 'last_model.pt')}")

    # 评估模型
    metrics = model.val()

    # 提取关键性能指标
    box_metrics = {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr,
        'F1 Score': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (
                                                                                                                                   metrics.box.mp + metrics.box.mr) > 0 else 0
    }

    # 提取每个类别的性能指标
    class_names = metrics.names
    class_metrics = []

    for i, name in enumerate(class_names):
        if metrics.box.maps is not None and len(metrics.box.maps) > i:
            # 安全地获取类别指标
            try:
                if hasattr(metrics.box.maps[i], '__len__') and len(metrics.box.maps[i]) >= 2:
                    class_metrics.append({
                        'Class': name,
                        'mAP50': metrics.box.maps[i][0],  # mAP50 for class
                        'mAP50-95': metrics.box.maps[i][1],  # mAP50-95 for class
                    })
                else:
                    # 如果maps[i]是标量，只使用mAP50
                    class_metrics.append({
                        'Class': name,
                        'mAP50': float(metrics.box.maps[i]),
                        'mAP50-95': 0.0,  # 默认值
                    })
            except (IndexError, TypeError, AttributeError) as e:
                print(f"警告: 无法获取类别 {name} 的指标: {e}")
                class_metrics.append({
                    'Class': name,
                    'mAP50': 0.0,
                    'mAP50-95': 0.0,
                })

    # 打印总体性能指标
    print("\n===== 总体性能指标 =====")
    print(tabulate(
        [(k, f"{v:.4f}") for k, v in box_metrics.items()],
        headers=["指标", "数值"],
        tablefmt="fancy_grid"
    ))

    # 打印类别性能指标（如果有）
    if class_metrics:
        print("\n===== 类别性能指标 =====")
        print(tabulate(
            [(m['Class'], f"{m['mAP50']:.4f}", f"{m['mAP50-95']:.4f}") for m in class_metrics],
            headers=["类别", "mAP50", "mAP50-95"],
            tablefmt="fancy_grid"
        ))

    # 保存性能指标到CSV文件
    metrics_df = pd.DataFrame(box_metrics, index=[0])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)

    if class_metrics:
        class_df = pd.DataFrame(class_metrics)
        class_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False)

    print(f"\n所有训练结果已保存至: {save_dir}")
    
    # ===== 训练完成后自动检测代表性图像 =====
    print("\n" + "="*60)
    print("🎯 开始检测代表性图像...")
    print("="*60)
    
    # 代表性图像路径
    representative_dir = os.path.join(os.getcwd(), 'representative_defect_images')
    
    # 改进检测结果路径 - 保存到runs目录下
    improved_results_dir = os.path.join(save_dir, 'improved_detection_results')
    
    # 检查代表性图像目录是否存在
    if not os.path.exists(representative_dir):
        print(f"❌ 代表性图像目录不存在: {representative_dir}")
        print("请确保代表性图像已正确复制到指定目录")
    else:
        # 创建检测结果保存目录
        detection_results_dir = os.path.join(save_dir, 'representative_detection_results')
        os.makedirs(detection_results_dir, exist_ok=True)
        
        # 创建改进检测结果目录
        os.makedirs(improved_results_dir, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        # 查找代表性图像
        for file in os.listdir(representative_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(representative_dir, file))
        
        if not image_files:
            print(f"❌ 在 {representative_dir} 中未找到图像文件")
        else:
            print(f"📸 找到 {len(image_files)} 张代表性图像")
            
            # 缺陷类别和颜色映射
            class_info = {
                'crazing': {'name': '龟裂', 'color': 'green'},
                'inclusion': {'name': '夹杂', 'color': 'red'},
                'patches': {'name': '斑块', 'color': 'blue'},
                'pitted_surface': {'name': '点蚀表面', 'color': 'orange'},
                'rolled-in_scale': {'name': '轧入氧化皮', 'color': 'yellow'},
                'scratches': {'name': '划痕', 'color': 'cyan'}
            }
            
            # 使用训练好的最佳模型进行检测
            best_model_path = os.path.join(save_dir, 'yolov8_training', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                print(f"🔧 使用训练好的最佳模型: {best_model_path}")
                trained_model = YOLO(best_model_path)
            else:
                print("⚠️  最佳模型不存在，使用原始模型")
                trained_model = model
            
            # 检测每张代表性图像
            detection_summary = []
            for i, image_path in enumerate(image_files):
                print(f"\n🔍 检测图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                try:
                    # 进行检测
                    results = trained_model(image_path, conf=0.25, verbose=False)
                    
                    # 保存检测结果到训练结果目录
                    output_name = f"detection_{os.path.splitext(os.path.basename(image_path))[0]}_trained.jpg"
                    output_path = os.path.join(detection_results_dir, output_name)
                    results[0].save(output_path)
                    
                    # 同时保存到改进检测结果目录
                    improved_output_name = f"improved_detection_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                    improved_output_path = os.path.join(improved_results_dir, improved_output_name)
                    results[0].save(improved_output_path)
                    
                    # 统计检测结果
                    if results[0].boxes is not None:
                        num_detections = len(results[0].boxes)
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                        
                        print(f"   ✅ 检测到 {num_detections} 个缺陷:")
                        for j, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                            if class_id < len(class_info):
                                class_names = list(class_info.keys())
                                class_name = class_names[class_id]
                                chinese_name = class_info[class_name]['name']
                                print(f"      - {chinese_name}: {conf:.1%}")
                        
                        detection_summary.append({
                            'image': os.path.basename(image_path),
                            'detections': num_detections,
                            'classes': [class_names[class_id] for class_id in class_ids if class_id < len(class_names)]
                        })
                    else:
                        print("   ⚠️  未检测到任何缺陷")
                        detection_summary.append({
                            'image': os.path.basename(image_path),
                            'detections': 0,
                            'classes': []
                        })
                    
                    print(f"   💾 结果已保存: {output_path}")
                    
                except Exception as e:
                    print(f"   ❌ 检测失败: {e}")
                    continue
            
            # 创建检测结果总结
            print(f"\n📊 代表性图像检测总结:")
            print("-" * 50)
            total_detections = sum(item['detections'] for item in detection_summary)
            print(f"总检测数量: {total_detections}")
            print(f"检测结果保存在: {detection_results_dir}")
            
            # 按类别统计
            all_classes = []
            for item in detection_summary:
                all_classes.extend(item['classes'])
            
            if all_classes:
                from collections import Counter
                class_counts = Counter(all_classes)
                print(f"\n各类缺陷检测统计:")
                for class_name, count in class_counts.items():
                    chinese_name = class_info.get(class_name, {}).get('name', class_name)
                    print(f"  - {chinese_name}: {count} 个")
            
            print(f"\n🎉 代表性图像检测完成！")
            print(f"📁 训练结果保存在: {detection_results_dir}")
            print(f"📁 改进检测结果保存在: {improved_results_dir}")
            print("="*60)
            
            # ===== 生成详细的训练指标报告 =====
            print("\n" + "="*60)
            print("📊 正在生成详细的训练指标报告...")
            print("="*60)
            
            try:
                # 1. 执行最终验证以收集详细指标
                print("✨ 正在执行最终验证以收集详细指标...")
                val_results = model.val()
                
                # 2. 提取总体性能指标
                mAP50 = val_results.box.map50
                mAP50_95 = val_results.box.map
                precision = val_results.box.mp
                recall = val_results.box.mr
                mAP75 = val_results.box.map75
                
                # 计算 F1 分数
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # 3. 获取模型基本信息
                model_info = model.info()
                params_million = model_info[0] / 1e6  # 转换为百万参数
                gflops = model_info[1]
                
                # 4. 获取每个类别的 mAP@0.5
                class_names = list(val_results.names.values())
                per_class_mAP50 = []
                
                # 安全地获取每个类别的指标
                if val_results.box.maps is not None:
                    for i in range(len(class_names)):
                        try:
                            if i < len(val_results.box.maps):
                                if hasattr(val_results.box.maps[i], '__len__') and len(val_results.box.maps[i]) >= 1:
                                    per_class_mAP50.append(float(val_results.box.maps[i][0]))
                                else:
                                    per_class_mAP50.append(float(val_results.box.maps[i]))
                            else:
                                per_class_mAP50.append(0.0)
                        except (IndexError, TypeError, AttributeError):
                            per_class_mAP50.append(0.0)
                else:
                    per_class_mAP50 = [0.0] * len(class_names)
                
                # 5. 格式化指标为字符串
                output_string = f"""--- 模型训练指标总结 ---

模型: YOLOv8n
类别: single-stage (单阶段)
参数 (M): {params_million:.2f}
GFLOPS: {gflops:.0f}

总体性能指标:
  mAP@0.5: {mAP50:.4f}
  mAP@0.75: {mAP75:.4f}
  mAP@0.5-0.95: {mAP50_95:.4f}
  精确率 (Precision): {precision:.4f}
  召回率 (Recall): {recall:.4f}
  F1 分数 (F1 Score): {f1_score:.4f}

各类别 mAP@0.5:
"""
                
                # 添加每个类别的指标
                for i, class_name in enumerate(class_names):
                    if i < len(per_class_mAP50):
                        output_string += f"  - {class_name.capitalize()}: {per_class_mAP50[i]:.4f}\n"
                    else:
                        output_string += f"  - {class_name.capitalize()}: N/A (mAP 不可用)\n"
                
                # 6. 保存指标到文件
                metrics_file_path = os.path.join(save_dir, 'training_metrics_summary.txt')
                
                with open(metrics_file_path, 'w', encoding='utf-8') as f:
                    f.write(output_string)
                
                print(f"✅ 训练指标已成功保存到: {metrics_file_path}")
                
                # 7. 在控制台显示关键指标
                print("\n🎯 关键性能指标:")
                print(f"   mAP@0.5: {mAP50:.4f}")
                print(f"   mAP@0.75: {mAP75:.4f}")
                print(f"   mAP@0.5-0.95: {mAP50_95:.4f}")
                print(f"   精确率: {precision:.4f}")
                print(f"   召回率: {recall:.4f}")
                print(f"   F1分数: {f1_score:.4f}")
                
                # 8. 显示各类别性能
                print("\n📈 各类别 mAP@0.5 性能:")
                for i, class_name in enumerate(class_names):
                    if i < len(per_class_mAP50):
                        print(f"   {class_name.capitalize()}: {per_class_mAP50[i]:.4f}")
                
            except Exception as e:
                print(f"❌ 生成训练指标时发生错误: {e}")
                print("继续执行其他任务...")
            
            # ===== 显示所有保存位置总结 =====
            print("\n" + "="*60)
            print("📁 所有训练结果保存位置总结")
            print("="*60)
            print(f"🏠 项目根目录: {os.getcwd()}")
            print(f"📂 训练结果主目录: {save_dir}")
            print(f"🤖 模型文件:")
            print(f"   - 最佳模型: {os.path.join(save_dir, 'yolov8_training', 'weights', 'best.pt')}")
            print(f"   - 最终模型: {os.path.join(save_dir, 'yolov8_training', 'weights', 'last.pt')}")
            print(f"   - 导出最佳模型: {os.path.join(save_dir, 'exported_models', 'best_model.pt')}")
            print(f"   - 导出最终模型: {os.path.join(save_dir, 'exported_models', 'last_model.pt')}")
            print(f"📊 性能指标文件:")
            print(f"   - 详细指标报告: {os.path.join(save_dir, 'training_metrics_summary.txt')}")
            print(f"   - CSV格式指标: {os.path.join(save_dir, 'metrics_summary.csv')}")
            print(f"   - 类别指标: {os.path.join(save_dir, 'class_metrics.csv')}")
            print(f"🖼️ 检测结果:")
            print(f"   - 代表性图像检测: {os.path.join(save_dir, 'representative_detection_results')}")
            print(f"   - 改进检测结果: {improved_results_dir}")
            print(f"📈 训练日志和图表: {os.path.join(save_dir, 'yolov8_training')}")
            print("="*60)