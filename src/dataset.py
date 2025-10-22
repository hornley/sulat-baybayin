import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(img_size=224, augment=False):
    """Return (train_transform, val_transform)"""
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform

    return train_transform, val_transform


class BaybayinDataset(Dataset):
    """Simple image dataset expecting a folder per-class.

    Directory structure:
        root/class_x/xxx.png
        root/class_y/123.jpg
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def make_dataloaders(root, batch_size=32, img_size=224, val_split=0.2, random_state=42, augment=False, min_count=1):
    """Scan root folder for classes, build train/val loaders with 80/20 split.

    Returns: train_loader, val_loader, class_names
    """
    classes = []
    image_paths = []
    labels = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            imgs = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if len(imgs) < min_count:
                # skip classes with fewer than min_count images
                continue
            classes.append(entry)
            for fname in sorted(imgs):
                image_paths.append(os.path.join(p, fname))
                labels.append(len(classes) - 1)

    if not classes:
        raise RuntimeError(f'No class subfolders found in {root}')

    # Validation / basic transform
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_transform, val_transform = get_transforms(img_size=img_size, augment=augment)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=random_state
    )

    train_ds = BaybayinDataset(train_paths, train_labels, transform=train_transform)
    val_ds = BaybayinDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, classes
