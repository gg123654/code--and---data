# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Adaptive Multi-Scale Defect-Aware Loss (AMSDAL) for NEU-DET Steel Surface Defect Detection

This loss function is specifically designed for steel surface defect detection tasks,
addressing the unique challenges of the NEU-DET dataset:
1. Small defect targets requiring adaptive scale weighting
2. Class imbalance and varying detection difficulty
3. High precision requirements for defect boundaries
4. Multi-scale feature fusion for better detection

Key Innovations:
- Adaptive Scale-Weighted Loss: Dynamically adjusts loss weights based on target size
- Class Difficulty-Aware Classification: Adapts to different defect class difficulties
- Boundary Precision Enhancement: Emphasizes boundary accuracy for defect detection
- Multi-Scale Consistency Loss: Ensures consistent predictions across scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import BboxLoss, DFLoss


class AdaptiveScaleWeightedBboxLoss(BboxLoss):
    """
    Enhanced bbox loss with adaptive scale weighting for small defect targets.
    Smaller targets receive higher weights to improve detection of tiny defects.
    """
    
    def __init__(self, reg_max=16, small_target_threshold=0.05, scale_weight_power=1.5):
        """
        Initialize adaptive scale-weighted bbox loss.
        
        Args:
            reg_max: Maximum regression value for DFL
            small_target_threshold: Threshold to identify small targets (relative to image size)
            scale_weight_power: Power factor for scale weighting (higher = more emphasis on small targets)
        """
        super().__init__(reg_max)
        self.small_target_threshold = small_target_threshold
        self.scale_weight_power = scale_weight_power
    
    def compute_adaptive_weights(self, target_bboxes, imgsz):
        """
        Compute adaptive weights based on target size.
        Smaller targets get higher weights.
        
        Args:
            target_bboxes: Ground truth bounding boxes (xyxy format)
            imgsz: Image size (h, w)
            
        Returns:
            Adaptive weights tensor
        """
        # Convert to xywh to get width and height
        target_xywh = xyxy2xywh(target_bboxes)
        areas = target_xywh[..., 2] * target_xywh[..., 3]  # width * height
        
        # Normalize by image area
        img_area = imgsz[0] * imgsz[1]
        normalized_areas = areas / img_area
        
        # Compute weights: smaller targets get higher weights
        # Using inverse relationship with power scaling
        base_weights = 1.0 / (normalized_areas + 1e-6)
        normalized_weights = base_weights / (base_weights.mean() + 1e-6)
        
        # Apply power scaling for more aggressive weighting
        adaptive_weights = normalized_weights ** self.scale_weight_power
        
        # Clamp to reasonable range to avoid extreme values
        adaptive_weights = torch.clamp(adaptive_weights, 0.5, 3.0)
        
        return adaptive_weights.unsqueeze(-1)
    
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, 
                target_scores, target_scores_sum, fg_mask, imgsz=None):
        """
        Forward pass with adaptive scale weighting.
        
        Args:
            imgsz: Image size for computing adaptive weights
        """
        # Compute adaptive weights based on target size
        if imgsz is not None:
            adaptive_weights = self.compute_adaptive_weights(
                target_bboxes[fg_mask] * imgsz[[1, 0, 1, 0]], imgsz
            )
        else:
            adaptive_weights = torch.ones(
                (fg_mask.sum(), 1), 
                device=target_bboxes.device, 
                dtype=target_bboxes.dtype
            )
        
        # Standard IoU loss with adaptive weighting
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1) * adaptive_weights
        
        # Add class-specific boost for crazing (class 0) and rolled-in_scale (class 4)
        if target_scores.shape[-1] > 0:  # Check if we have class information
            target_classes = target_scores[fg_mask].argmax(dim=-1)
            class_boost = torch.ones_like(weight.squeeze(-1))
            # Boost crazing (class 0)
            crazing_mask = (target_classes == 0)
            if crazing_mask.any():
                class_boost[crazing_mask] *= 2.0  # 2x boost for crazing
            # Boost rolled-in_scale (class 4)
            rolled_mask = (target_classes == 4)
            if rolled_mask.any():
                class_boost[rolled_mask] *= 2.0  # 2x boost for rolled-in_scale
            weight = weight * class_boost.unsqueeze(-1)
        
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / (weight.sum() + 1e-6)
        
        # DFL loss with adaptive weighting
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), 
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / (weight.sum() + 1e-6)
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        
        return loss_iou, loss_dfl


class ClassDifficultyAwareLoss(nn.Module):
    """
    Classification loss with class difficulty awareness.
    Adapts to different detection difficulties across defect classes.
    """
    
    def __init__(self, num_classes=6, initial_difficulty=None, adaptation_rate=0.01, class_weights=None):
        """
        Initialize class difficulty-aware loss.
        
        Args:
            num_classes: Number of defect classes
            initial_difficulty: Initial difficulty weights for each class (None = equal)
            adaptation_rate: Rate at which difficulty weights adapt during training
            class_weights: Optional static class weights to boost specific classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate
        
        # Initialize class difficulty weights (learnable)
        if initial_difficulty is None:
            # Default: slightly higher weight for classes that might be harder
            # Based on typical defect characteristics in NEU-DET
            initial_difficulty = torch.ones(num_classes) * 1.0
            # You can adjust these based on your observations
            # e.g., if 'crazing' is harder to detect, give it higher weight
        
        self.register_buffer('class_difficulty', initial_difficulty.clone())
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
        # Track classification errors per class for adaptive adjustment
        self.register_buffer('class_error_ema', torch.zeros(num_classes))
        self.ema_momentum = 0.9

        # Static class weights (e.g., boost crazing=0 and rolled-in_scale=4)
        # 根据训练结果，大幅提升crazing和rolled-in_scale的权重
        if class_weights is None:
            class_weights = torch.tensor([
                6.0,  # crazing: 从4.5->6.0，极大幅度提升（当前0.691，目标>0.75）
                1.2,  # inclusion: 保持1.2（性能已很好0.806）
                1.0,  # patches: 保持1.0（性能已很好0.914）
                1.0,  # pitted_surface: 保持1.0（性能已很好0.849）
                5.5,  # rolled-in_scale: 从4.0->5.5，极大幅度提升（当前0.581，目标>0.65）
                1.1,  # scratches: 保持1.1（性能已很好0.878）
            ])
        self.register_buffer('class_weights', class_weights.float())
    
    def update_difficulty(self, pred_scores, target_scores, class_indices):
        """
        Update class difficulty weights based on classification errors.
        
        Args:
            pred_scores: Predicted class scores
            target_scores: Target class scores
            class_indices: Class indices for each target
        """
        with torch.no_grad():
            # Compute classification errors per class
            pred_probs = pred_scores.sigmoid()
            errors = (pred_probs - target_scores).abs()
            
            # Aggregate errors by class
            for cls_idx in range(self.num_classes):
                cls_mask = (class_indices == cls_idx)
                if cls_mask.any():
                    cls_error = errors[cls_mask].mean().item()
                    # Update EMA
                    self.class_error_ema[cls_idx] = (
                        self.ema_momentum * self.class_error_ema[cls_idx] + 
                        (1 - self.ema_momentum) * cls_error
                    )
                    # Adjust difficulty: higher error -> higher difficulty weight
                    error_ratio = self.class_error_ema[cls_idx] / (self.class_error_ema.mean() + 1e-6)
                    self.class_difficulty[cls_idx] = (
                        (1 - self.adaptation_rate) * self.class_difficulty[cls_idx] +
                        self.adaptation_rate * (1.0 + error_ratio * 0.5)
                    )
                    # Clamp to reasonable range
                    self.class_difficulty[cls_idx] = torch.clamp(
                        self.class_difficulty[cls_idx], 0.5, 2.0
                    )
    
    def forward(self, pred_scores, target_scores, class_indices=None):
        """
        Compute classification loss with class difficulty weighting.
        
        Args:
            pred_scores: Predicted class scores
            target_scores: Target class scores
            class_indices: Class indices for weighting (optional)
        """
        # Standard BCE loss
        loss = self.bce(pred_scores, target_scores)
        
        # Apply class difficulty weighting if class indices provided
        if class_indices is not None:
            # Get difficulty weights for each sample
            difficulty_weights = self.class_difficulty[class_indices.long()]
            loss = loss * difficulty_weights.unsqueeze(-1)
            # Apply static class weights
            loss = loss * self.class_weights[class_indices.long()].unsqueeze(-1)
        
        return loss


class BoundaryPrecisionLoss(nn.Module):
    """
    Boundary precision enhancement loss for defect detection.
    Emphasizes accurate boundary localization which is critical for defect analysis.
    """
    
    def __init__(self, boundary_weight=2.0, boundary_threshold=0.1):
        """
        Initialize boundary precision loss.
        
        Args:
            boundary_weight: Weight factor for boundary regions
            boundary_threshold: Threshold to define boundary region (relative to bbox size)
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_threshold = boundary_threshold
    
    def compute_boundary_mask(self, pred_bboxes, target_bboxes):
        """
        Compute mask for boundary regions where precision is critical.
        
        Args:
            pred_bboxes: Predicted bounding boxes
            target_bboxes: Target bounding boxes
            
        Returns:
            Boundary importance weights
        """
        # Compute IoU
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=False)
        
        # Higher weight for boxes with lower IoU (boundary regions need more attention)
        # Also weight by how close we are to the boundary
        boundary_importance = (1.0 - iou) * self.boundary_weight
        
        return boundary_importance.unsqueeze(-1)
    
    def forward(self, pred_bboxes, target_bboxes):
        """
        Compute boundary precision loss.
        
        Returns:
            Boundary precision loss component
        """
        boundary_weights = self.compute_boundary_mask(pred_bboxes, target_bboxes)
        
        # Compute distance-based boundary loss
        # Distance between predicted and target box centers
        pred_center = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
        target_center = (target_bboxes[..., :2] + target_bboxes[..., 2:]) / 2
        center_distance = torch.norm(pred_center - target_center, dim=-1)
        
        # Distance between box sizes
        pred_size = pred_bboxes[..., 2:] - pred_bboxes[..., :2]
        target_size = target_bboxes[..., 2:] - target_bboxes[..., :2]
        size_distance = torch.norm(pred_size - target_size, dim=-1)
        
        # Combined boundary loss
        boundary_loss = (center_distance + size_distance) * boundary_weights.squeeze(-1)
        
        return boundary_loss.mean()


class AMSDALoss:
    """
    Adaptive Multi-Scale Defect-Aware Loss (AMSDAL) for NEU-DET dataset.
    
    This loss function combines multiple innovative components:
    1. Adaptive scale-weighted bbox loss for small defect detection
    2. Class difficulty-aware classification loss
    3. Boundary precision enhancement loss
    4. Multi-scale consistency regularization
    """
    
    def __init__(self, model, tal_topk=10, 
                 small_target_threshold=0.05,
                 scale_weight_power=1.5,
                 boundary_weight=2.0,
                 class_adaptation_rate=0.01,
                 consistency_weight=0.1):
        """
        Initialize AMSDAL loss.
        
        Args:
            model: YOLO model (must be de-paralleled)
            tal_topk: Top-k for Task Aligned Assigner
            small_target_threshold: Threshold for small target detection
            scale_weight_power: Power factor for scale weighting
            boundary_weight: Weight for boundary precision loss
            class_adaptation_rate: Rate for class difficulty adaptation
            consistency_weight: Weight for multi-scale consistency loss
        """
        device = next(model.parameters()).device
        h = model.args
        
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.consistency_weight = consistency_weight
        
        # Initialize components
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = AdaptiveScaleWeightedBboxLoss(
            reg_max=m.reg_max,
            small_target_threshold=small_target_threshold,
            scale_weight_power=scale_weight_power
        ).to(device)
        self.cls_loss = ClassDifficultyAwareLoss(
            num_classes=self.nc,
            adaptation_rate=class_adaptation_rate
        ).to(device)
        self.boundary_loss = BoundaryPrecisionLoss(
            boundary_weight=boundary_weight
        ).to(device)
        
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bounding boxes."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def compute_consistency_loss(self, pred_bboxes_list, pred_scores_list):
        """
        Compute multi-scale consistency loss.
        Encourages consistent predictions across different scales.
        """
        if len(pred_bboxes_list) < 2:
            return torch.tensor(0.0, device=self.device)
        
        consistency_loss = 0.0
        n_pairs = 0
        
        # Compare predictions across adjacent scales
        for i in range(len(pred_bboxes_list) - 1):
            bbox1 = pred_bboxes_list[i]
            bbox2 = pred_bboxes_list[i + 1]
            score1 = pred_scores_list[i]
            score2 = pred_scores_list[i + 1]
            
            # Resize to same size for comparison (use smaller size)
            if bbox1.shape[1] != bbox2.shape[1]:
                min_size = min(bbox1.shape[1], bbox2.shape[1])
                bbox1 = bbox1[:, :min_size, :]
                bbox2 = bbox2[:, :min_size, :]
                score1 = score1[:, :min_size, :]
                score2 = score2[:, :min_size, :]
            
            # Consistency in bbox predictions (for overlapping regions)
            bbox_diff = torch.abs(bbox1 - bbox2).mean()
            
            # Consistency in class predictions
            score_diff = torch.abs(score1.sigmoid() - score2.sigmoid()).mean()
            
            consistency_loss += bbox_diff + score_diff
            n_pairs += 1
        
        return consistency_loss / (n_pairs + 1e-6)
    
    def __call__(self, preds, batch):
        """
        Compute AMSDAL loss.
        
        Returns:
            Total loss and individual loss components
        """
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, boundary, consistency
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # Process predictions
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # Task-aligned assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification loss with class difficulty awareness
        # class_indices derived from target_scores for weighting
        class_indices = target_scores.argmax(dim=-1)
        cls_loss_raw = self.cls_loss(pred_scores, target_scores.to(dtype), class_indices)
        
        # Get class indices for difficulty weighting
        if fg_mask.any():
            target_classes = target_scores.argmax(dim=-1)
            cls_loss_raw = cls_loss_raw * target_scores  # Weight by target scores
            self.cls_loss.update_difficulty(
                pred_scores[fg_mask], 
                target_scores[fg_mask], 
                target_classes[fg_mask]
            )
        
        loss[1] = cls_loss_raw.sum() / target_scores_sum
        
        # Bbox loss with adaptive scale weighting
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            
            # Adaptive scale-weighted bbox loss
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz=imgsz
            )
            
            # Boundary precision loss
            # Use normalized coordinates for boundary loss
            loss[3] = self.boundary_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask]
            )
        
        # Multi-scale consistency loss
        if len(feats) > 1:
            pred_bboxes_list = []
            pred_scores_list = []
            for idx, feat in enumerate(feats):
                feat_pred_distri, feat_pred_scores = feat.view(feat.shape[0], self.no, -1).split(
                    (self.reg_max * 4, self.nc), 1
                )
                feat_pred_scores = feat_pred_scores.permute(0, 2, 1).contiguous()
                feat_pred_distri = feat_pred_distri.permute(0, 2, 1).contiguous()
                feat_anchor_points, _ = make_anchors([feat], [self.stride[idx]], 0.5)
                feat_pred_bboxes = self.bbox_decode(feat_anchor_points, feat_pred_distri)
                pred_bboxes_list.append(feat_pred_bboxes)
                pred_scores_list.append(feat_pred_scores)
            
            loss[4] = self.compute_consistency_loss(pred_bboxes_list, pred_scores_list)
        
        # Apply hyperparameter weights
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= 0.5  # boundary gain (custom)
        loss[4] *= self.consistency_weight  # consistency gain
        
        # Return total loss and individual components
        loss_items = loss.detach()
        return loss.sum() * batch_size, loss_items

