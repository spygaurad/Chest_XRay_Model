import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class DiceLoss(nn.Module):

    def __init__(self, num_classes=8):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.classes = 3
        self.ignore_index = None
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc




class JaccardScore(nn.Module):

    def __init__(self):
        super(JaccardScore, self).__init__()
    
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = torch.logical_and(y_true, y_pred)
        union = torch.logical_or(y_true, y_pred)
        iou_score = torch.sum(intersection) / torch.sum(union)  
        return iou_score




class MixedLoss(nn.Module):
  def __init__(self, alpha, beta):
    super(MixedLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta

  def forward(self, y_pred, y_true):
    # y_pred and y_true are of shape (batch_size, channels, height, width)
    # compute the MS-SSIM loss
    msssim_loss = 1 - ms_ssim(y_pred, y_true)
    # compute the L2 loss
    l2_loss = nn.MSELoss()(y_pred, y_true)
    # return the mixed loss
    return self.alpha*msssim_loss + self.beta*l2_loss




class YOLOLoss(nn.Module):
    def __init__(self, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        batch_size, S, S, _ = predictions.shape
        B = 1  # Assuming a single bounding box prediction per grid cell

        # Extract target values
        target_coords = targets[..., :4].view(batch_size, S, S, B, 4)
        target_confs = targets[..., 4:5].view(batch_size, S, S, B, 1)
        target_classes = targets[..., 5:].view(batch_size, S, S, B, self.num_classes)

        # Extract prediction values
        pred_coords = predictions[..., :4].view(batch_size, S, S, B, 4)
        pred_confs = predictions[..., 4:5].view(batch_size, S, S, B, 1)
        pred_classes = predictions[..., 5:].view(batch_size, S, S, B, self.num_classes)

        # Calculate localization loss (MSE between predicted and target bounding box coordinates)
        coord_loss = self.mse_loss(pred_coords, target_coords)

        # Calculate confidence loss for object detection
        obj_mask = target_confs.squeeze(-1)  # Mask for cells containing objects
        noobj_mask = 1.0 - obj_mask  # Mask for cells without objects

        # Object confidence loss (MSE between predicted and target object confidence scores)
        obj_conf_loss = self.mse_loss(pred_confs[obj_mask], target_confs[obj_mask])

        # No-object confidence loss (MSE between predicted and target object confidence scores)
        noobj_conf_loss = self.mse_loss(pred_confs[noobj_mask], target_confs[noobj_mask])

        # Calculate class loss (binary cross-entropy loss between predicted and target class probabilities)
        class_loss = self.bce_loss(pred_classes[obj_mask], target_classes[obj_mask])

        # Calculate total YOLO loss
        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )

        return total_loss


class MultiLabelAUROC(nn.Module):
    def __init__(self):
        super(MultiLabelAUROC, self).__init__()

    def forward(self, y_gt, y_pred):
        auroc = []
        gt_np = y_gt.cpu().numpy()
        pred_np = y_pred.cpu().numpy()
        assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
        for i in range(gt_np.shape[1]):
            try:
                auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            except ValueError:
                pass
        return torch.tensor(auroc)


class MultiLabelAccuracy(nn.Module):
    def __init__(self):
        super(MultiLabelAccuracy, self).__init__()

    def forward(self, y_gt, y_pred):
        acc = []
        gt_np = y_gt.cpu().numpy()
        pred_np = y_pred.cpu().numpy()
        assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
        for i in range(gt_np.shape[1]):
            acc.append(accuracy_score(gt_np[:, i], np.where(pred_np[:, i] >= 0.5, 1, 0)))
        return torch.tensor(acc)


class MultiLabelF1(nn.Module):
    def __init__(self):
        super(MultiLabelF1, self).__init__()

    def forward(self, y_gt, y_pred):
        f1_out = []
        gt_np = y_gt.cpu().numpy()
        pred_np = y_pred.cpu().numpy()
        assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
        for i in range(gt_np.shape[1]):
            f1_out.append(f1_score(gt_np[:, i], np.where(pred_np[:, i] >= 0.5, 1, 0)))
        return torch.tensor(f1_out)


class MultiLabelPrecisionRecall(nn.Module):
    def __init__(self):
        super(MultiLabelPrecisionRecall, self).__init__()

    def forward(self, y_gt, y_pred):
        precision_out = []
        recall_out = []
        gt_np = y_gt.cpu().numpy()
        pred_np = y_pred.cpu().numpy()
        assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
        for i in range(gt_np.shape[1]):
            p = precision_recall_fscore_support(gt_np[:, i], np.where(pred_np[:, i] >= 0.5, 1, 0), average='binary')
            precision_out.append(p[0])
            recall_out.append(p[1])
        return torch.tensor(precision_out), torch.tensor(recall_out)