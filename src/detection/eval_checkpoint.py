import argparse
import torch
import os
from collections import defaultdict
from src.detection.dataset import BBoxDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model_from_ckpt(ckpt_path, device='cpu'):
    ck = torch.load(ckpt_path, map_location=device)
    model_state = ck.get('model_state', ck)
    classes = ck.get('classes', [])
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, classes


def iou_matrix(a, b):
    # a: Nx4, b: Mx4 (tensors)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def evaluate_ckpt(ckpt, data_root, ann_file, batch=1, thresh=0.5, iou_thresh=0.5, device='cpu', out_dir=None):
    ds = BBoxDataset(data_root, ann_file=ann_file, transforms=None)
    if not ds:
        print('Dataset empty or not found')
        return
    model, classes = load_model_from_ckpt(ckpt, device=device)

    # stats per class
    stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in classes}
    total_images = 0
    compiled_preds = []

    # simple loader (no extra workers here)
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs_t = [img.to(device) for img in imgs]
            outputs = model(imgs_t)
            for out, tgt, img_path in zip(outputs, targets, [t.get('image_id', None) or '' for t in targets]):
                # targets: dict with 'boxes' tensor and 'labels'
                gt_boxes = tgt.get('boxes', torch.zeros((0,4))).to(device)
                gt_labels = tgt.get('labels', torch.zeros((0,), dtype=torch.int64)).to(device)
                pred_boxes = out.get('boxes', torch.zeros((0,4))).to(device)
                pred_labels = out.get('labels', torch.zeros((0,), dtype=torch.int64)).to(device)
                pred_scores = out.get('scores', torch.zeros((0,))).to(device)

                # filter by score
                keep = pred_scores >= thresh
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                # record compiled predictions for optional CSV
                if out_dir is not None:
                    for pb, pl, ps in zip(pred_boxes.cpu().tolist(), pred_labels.cpu().tolist(), pred_scores.cpu().tolist()):
                        x1,y1,x2,y2 = pb
                        lbl = classes[pl - 1] if pl > 0 and pl-1 < len(classes) else str(int(pl))
                        compiled_preds.append({'image': str(tgt.get('image_id', '')), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': lbl, 'score': ps})

                # matching: for each gt, find a pred with same label and iou >= iou_thresh
                if gt_boxes.numel() == 0:
                    # all preds are false positives for this image
                    for pl in pred_labels.cpu().tolist():
                        cname = classes[pl - 1] if pl > 0 and pl-1 < len(classes) else str(int(pl))
                        if cname in stats:
                            stats[cname]['fp'] += 1
                    total_images += 1
                    continue

                if pred_boxes.numel() == 0:
                    # all gts are false negatives
                    for gl in gt_labels.cpu().tolist():
                        cname = classes[gl - 1] if gl > 0 and gl-1 < len(classes) else str(int(gl))
                        if cname in stats:
                            stats[cname]['fn'] += 1
                    total_images += 1
                    continue

                ious = iou_matrix(gt_boxes, pred_boxes)
                # for each gt, find best pred of same label
                matched_pred = set()
                for gi in range(gt_boxes.shape[0]):
                    gl = int(gt_labels[gi].item())
                    cname = classes[gl - 1] if gl > 0 and gl-1 < len(classes) else str(gl)
                    # candidates with same label
                    candidate_idxs = [pi for pi, pl in enumerate(pred_labels.tolist()) if int(pl) == gl]
                    if not candidate_idxs:
                        stats[cname]['fn'] += 1
                        continue
                    ious_g = ious[gi]
                    # mask non-candidates
                    mask = torch.zeros_like(ious_g)
                    for ci in candidate_idxs:
                        mask[ci] = 1.0
                    ious_masked = ious_g * mask
                    best_iou, best_idx = torch.max(ious_masked, dim=0)
                    if best_iou.item() >= iou_thresh and best_idx.item() not in matched_pred:
                        stats[cname]['tp'] += 1
                        matched_pred.add(int(best_idx.item()))
                    else:
                        stats[cname]['fn'] += 1

                # any pred not matched is a false positive
                for pi in range(pred_boxes.shape[0]):
                    if pi not in matched_pred:
                        pl = int(pred_labels[pi].item())
                        cname = classes[pl - 1] if pl > 0 and pl-1 < len(classes) else str(pl)
                        if cname in stats:
                            stats[cname]['fp'] += 1

                total_images += 1

    # print summary
    print(f'Evaluated {total_images} images')
    sums = {'tp':0,'fp':0,'fn':0}
    for c, v in stats.items():
        sums['tp'] += v['tp']
        sums['fp'] += v['fp']
        sums['fn'] += v['fn']

    print('Overall: TP=%d FP=%d FN=%d' % (sums['tp'], sums['fp'], sums['fn']))
    print('Per-class precision/recall:')
    for c, v in stats.items():
        tp = v['tp']; fp = v['fp']; fn = v['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f'  {c}: TP={tp} FP={fp} FN={fn}  Precision={prec:.3f} Recall={rec:.3f}')

    # write compiled predictions CSV if requested
    if out_dir is not None and compiled_preds:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'compiled_annotations_eval.csv')
        import csv as _csv
        with open(csv_path, 'w', newline='', encoding='utf8') as cf:
            w = _csv.writer(cf)
            w.writerow(['image_path','x1','y1','x2','y2','label','confidence_score'])
            for r in compiled_preds:
                w.writerow([r['image'], r['x1'], r['y1'], r['x2'], r['y2'], r['label'], f"{r['score']:.6f}"])
        print('Wrote compiled predictions to', csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', required=True, help='Image root folder')
    parser.add_argument('--ann', required=True, help='CSV or COCO annotations file for evaluation')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out', default=None, help='Optional output folder to write compiled predictions')
    args = parser.parse_args()
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_ckpt(args.ckpt, args.data, args.ann, batch=args.batch, thresh=args.thresh, iou_thresh=args.iou, device=device, out_dir=args.out)


if __name__ == '__main__':
    main()
