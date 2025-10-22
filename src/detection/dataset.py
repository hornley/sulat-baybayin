import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset


class BBoxDataset(Dataset):
    """Loads bounding box annotations from a CSV with columns: image_file,xmin,ymin,xmax,ymax,label"""

    def __init__(self, csv_path, img_dir, transforms=None):
        self.records = []
        self.img_dir = img_dir
        self.transforms = transforms
        with open(csv_path, newline='') as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                # support both header and no-header formats: assume first row is header if non-numeric in coords
                try:
                    xmin = float(row[1])
                except Exception:
                    # skip header-like rows
                    continue
                img_file = row[0]
                xmin, ymin, xmax, ymax = map(float, row[1:5])
                label = row[5] if len(row) > 5 else 'object'
                self.records.append((img_file, xmin, ymin, xmax, ymax, label))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_file, xmin, ymin, xmax, ymax, label = self.records[idx]
        path = os.path.join(self.img_dir, img_file)
        img = Image.open(path).convert('RGB')
        w, h = img.size
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # single-class detection placeholder
        target = {'boxes': boxes, 'labels': labels}
        if self.transforms:
            img = self.transforms(img)
        return img, target
