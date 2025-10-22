import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

class BBoxDataset(Dataset):
    """
    Loader for simple bbox annotations.

    Supported annotation formats:
      - CSV: each line -> image_path,x1,y1,x2,y2,label
      - COCO-style JSON: fields 'images' and 'annotations' (basic support)

    Returns samples compatible with torchvision detection models:
      image (Tensor HxWxC), target dict with keys: boxes, labels, image_id, area, iscrowd
    """
    def __init__(self, root, ann_file=None, transforms=None, classes=None):
        self.root = root
        self.transforms = transforms
        self.samples = []  # list of (img_path, [bboxes], [labels])
        self.classes = classes or []
        if ann_file is None:
            # assume root contains images only and no annotations
            imgs = [f for f in os.listdir(root) if f.lower().endswith(EXTS)]
            for im in imgs:
                self.samples.append((os.path.join(root, im), [], []))
        else:
            if ann_file.lower().endswith('.csv'):
                self._load_csv(ann_file)
            elif ann_file.lower().endswith('.json'):
                self._load_coco(ann_file)
            else:
                raise RuntimeError('Unsupported annotation format: ' + ann_file)
        # build class -> idx map
        if not self.classes:
            # collect classes from samples
            classes = set()
            for _,_,labels in self.samples:
                classes.update(labels)
            classes = sorted(list(classes))
            self.classes = classes
        self.class_to_idx = {c:i+1 for i,c in enumerate(self.classes)}  # 0 reserved for background

    def _load_csv(self, path):
        # CSV format: image_path,x1,y1,x2,y2,label
        # multiple rows can refer to same image
        entries = {}
        with open(path,'r',encoding='utf8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=[p.strip() for p in line.split(',')]
                if len(parts) < 6: continue
                img, x1,y1,x2,y2,label = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[5]
                # Resolve image path robustly:
                # - if CSV provides absolute path, use it
                # - if root + img exists, use that (common case)
                # - if raw img path exists (CSV already included root or is correct relative path), use it
                # - otherwise fall back to joining root and img
                if os.path.isabs(img):
                    imgpath = img
                else:
                    joined = os.path.join(self.root, img)
                    if os.path.exists(joined):
                        imgpath = joined
                    elif os.path.exists(img):
                        imgpath = img
                    else:
                        imgpath = joined
                if imgpath not in entries:
                    entries[imgpath] = {'boxes':[], 'labels':[]}
                entries[imgpath]['boxes'].append([x1,y1,x2,y2])
                entries[imgpath]['labels'].append(label)
        for k,v in entries.items():
            self.samples.append((k, v['boxes'], v['labels']))

    def _load_coco(self, path):
        data = json.load(open(path,'r',encoding='utf8'))
        imgs = {img['id']: img for img in data.get('images', [])}
        anns_by_img = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            bbox = ann.get('bbox', ann.get('box', []))
            # COCO bbox format is [x,y,width,height]
            if bbox and len(bbox) == 4:
                x,y,w,h = bbox
                box = [x, y, x+w, y+h]
            else:
                continue
            cat = ann.get('category_id')
            if isinstance(cat, int):
                label = str(cat)
            else:
                label = str(cat)
            im = imgs.get(img_id)
            if not im: continue
            imgname = im.get('file_name')
            imgpath = imgname if os.path.isabs(imgname) else os.path.join(self.root, imgname)
            if imgpath not in anns_by_img:
                anns_by_img[imgpath] = {'boxes':[], 'labels':[]}
            anns_by_img[imgpath]['boxes'].append(box)
            anns_by_img[imgpath]['labels'].append(label)
        for k,v in anns_by_img.items():
            self.samples.append((k, v['boxes'], v['labels']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w,h = img.size
        # convert to tensors
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels_idx = [self.class_to_idx.get(l, 0) for l in labels]
        labels_t = torch.tensor(labels_idx, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes_t
        target['labels'] = labels_t
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes_t[:,3] - boxes_t[:,1]) * (boxes_t[:,2] - boxes_t[:,0]) if boxes else torch.zeros((0,))
        target['iscrowd'] = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
        if self.transforms:
            img, target = self.transforms(img, target)
        # torchvision detection expects image as tensor
        from torchvision.transforms import ToTensor
        img = ToTensor()(img)
        return img, target

if __name__ == '__main__':
    # quick smoke test
    ds = BBoxDataset('data', ann_file='annotations.csv')
    print(len(ds))
    im,t = ds[0]
    print(im.shape, t.keys())
