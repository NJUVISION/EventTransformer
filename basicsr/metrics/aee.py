def calculate_AEE(pred_flow, gt_flow, mask):
    flow_mag = pred_flow.pow(2).sum(1).sqrt()
    error = (pred_flow - gt_flow).pow(2).sum(1).sqrt()
    error = error * mask[:, 0]
    AEE = error.sum() / mask.sum()
    outlier = (error > 3) * (error > 0.05 * flow_mag)
    outlier = outlier.sum() / mask.sum()
    return AEE, outlier
