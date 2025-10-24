# classification dataset (moved from src/dataset.py)
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.shared.augmentations import overlay_paper_texture_pil, overlay_paper_lines_pil, apply_lighting_pil


class PaperAugmentation:
    """Apply paper texture, lines, and lighting augmentations to PIL images.
    
    This wraps the detection pipeline's paper augmentation system for use in
    classification training, providing realistic paper backgrounds and lighting.
    """
    
    def __init__(
        self,
        # paper texture augmentation
        paper_prob=0.0,
        paper_type_probs=None,  # [white, yellow-paper, dotted] probabilities
        paper_texture_probs=None,  # [plain, grainy, crumpled] probabilities
        paper_strength_min=0.2,
        paper_strength_max=0.4,
        paper_yellow_strength_min=0.3,
        paper_yellow_strength_max=0.5,
        crumple_strength_min=1.0,
        crumple_strength_max=3.0,
        crumple_mesh_overlap=2,
        # paper lines augmentation
        lines_prob=0.0,
        line_spacing_min=24,
        line_spacing_max=32,
        line_opacity_min=30,
        line_opacity_max=60,
        line_thickness_min=1,
        line_thickness_max=2,
        line_jitter_min=1,
        line_jitter_max=3,
        line_color=(0, 0, 0),
        # lighting augmentation
        lighting_prob=0.0,
        lighting_modes=None,  # ['normal', 'bright', 'dim', 'shadows']
        brightness_jitter=0.03,
        contrast_jitter=0.03,
        shadow_intensity_min=0.0,
        shadow_intensity_max=0.3,
        # dotted paper options
        dot_size=1,
        dot_opacity=50,
        dot_spacing=18,
    ):
        self.paper_prob = paper_prob
        self.paper_type_probs = paper_type_probs or [0.6, 0.2, 0.2]  # white, yellow, dotted
        self.paper_texture_probs = paper_texture_probs or [0.5, 0.3, 0.2]  # plain, grainy, crumpled
        self.paper_strength_min = paper_strength_min
        self.paper_strength_max = paper_strength_max
        self.paper_yellow_strength_min = paper_yellow_strength_min
        self.paper_yellow_strength_max = paper_yellow_strength_max
        self.crumple_strength_min = crumple_strength_min
        self.crumple_strength_max = crumple_strength_max
        self.crumple_mesh_overlap = crumple_mesh_overlap
        
        self.lines_prob = lines_prob
        self.line_spacing_min = line_spacing_min
        self.line_spacing_max = line_spacing_max
        self.line_opacity_min = line_opacity_min
        self.line_opacity_max = line_opacity_max
        self.line_thickness_min = line_thickness_min
        self.line_thickness_max = line_thickness_max
        self.line_jitter_min = line_jitter_min
        self.line_jitter_max = line_jitter_max
        self.line_color = line_color
        
        self.lighting_prob = lighting_prob
        self.lighting_modes = lighting_modes or ['normal', 'bright', 'dim', 'shadows']
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.shadow_intensity_min = shadow_intensity_min
        self.shadow_intensity_max = shadow_intensity_max
        
        self.dot_size = dot_size
        self.dot_opacity = dot_opacity
        self.dot_spacing = dot_spacing
        
        self.paper_types = ['white', 'yellow-paper', 'dotted']
        self.paper_textures = ['plain', 'grainy', 'crumpled']
    
    def __call__(self, img):
        """Apply augmentations to a PIL Image and return PIL Image."""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Apply paper texture augmentation
        if self.paper_prob > 0 and random.random() < self.paper_prob:
            paper_type = random.choices(self.paper_types, weights=self.paper_type_probs, k=1)[0]
            paper_texture = random.choices(self.paper_textures, weights=self.paper_texture_probs, k=1)[0]
            
            # Random strength parameters
            paper_strength = random.uniform(self.paper_strength_min, self.paper_strength_max)
            paper_yellow_strength = random.uniform(self.paper_yellow_strength_min, self.paper_yellow_strength_max)
            crumple_strength = random.uniform(self.crumple_strength_min, self.crumple_strength_max)
            
            # Line parameters for yellow-paper (always draws lines)
            line_spacing = random.randint(self.line_spacing_min, self.line_spacing_max)
            line_opacity = random.randint(self.line_opacity_min, self.line_opacity_max)
            line_thickness = random.randint(self.line_thickness_min, self.line_thickness_max)
            line_jitter = random.randint(self.line_jitter_min, self.line_jitter_max)
            
            img = overlay_paper_texture_pil(
                img,
                paper_type=paper_type,
                paper_texture=paper_texture,
                line_color=self.line_color,
                line_opacity=line_opacity,
                line_spacing=line_spacing,
                line_thickness=line_thickness,
                line_jitter=line_jitter,
                paper_strength=paper_strength,
                paper_yellow_strength=paper_yellow_strength,
                crumple_strength=crumple_strength,
                crumple_mesh_overlap=self.crumple_mesh_overlap,
                dot_size=self.dot_size,
                dot_opacity_override=self.dot_opacity,
                dot_uniform=True,  # always uniform for classification
                dot_spacing=self.dot_spacing
            )
        
        # Apply ruled lines independently (for white/dotted paper or when paper texture not applied)
        # Note: yellow-paper already has lines from overlay_paper_texture_pil
        if self.lines_prob > 0 and random.random() < self.lines_prob:
            line_spacing = random.randint(self.line_spacing_min, self.line_spacing_max)
            line_opacity = random.randint(self.line_opacity_min, self.line_opacity_max)
            line_thickness = random.randint(self.line_thickness_min, self.line_thickness_max)
            line_jitter = random.randint(self.line_jitter_min, self.line_jitter_max)
            
            img = overlay_paper_lines_pil(
                img,
                line_color=self.line_color,
                line_opacity=line_opacity,
                line_spacing=line_spacing,
                line_thickness=line_thickness,
                jitter=line_jitter
            )
        
        # Apply lighting variations
        if self.lighting_prob > 0 and random.random() < self.lighting_prob:
            mode = random.choice(self.lighting_modes)
            shadow_intensity = random.uniform(self.shadow_intensity_min, self.shadow_intensity_max)
            
            img = apply_lighting_pil(
                img,
                mode=mode,
                brightness_jitter=self.brightness_jitter,
                contrast_jitter=self.contrast_jitter,
                shadow_intensity=shadow_intensity
            )
        
        # Convert back to RGB for torchvision transforms
        return img.convert('RGB')


def get_transforms(img_size=224, augment=False, paper_aug=None):
    """Return (train_transform, val_transform)
    
    Args:
        img_size: Target image size
        augment: Enable standard geometric/color augmentations
        paper_aug: PaperAugmentation instance for paper/lighting effects (optional)
    """
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if augment or paper_aug is not None:
        transform_list = []
        
        # Add paper augmentation first (PIL → PIL) if provided
        if paper_aug is not None:
            transform_list.append(paper_aug)
        
        # Add standard geometric augmentations if enabled
        if augment:
            transform_list.extend([
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        else:
            # No geometric augmentation, just resize
            transform_list.append(transforms.Resize((img_size, img_size)))
        
        # Add tensor conversion and normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_transform = transforms.Compose(transform_list)
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


def make_dataloaders(root, batch_size=32, img_size=224, val_split=0.2, random_state=42, augment=False, min_count=1, paper_aug=None):
    """Scan root folder for classes, build train/val loaders with 80/20 split.

    Args:
        root: Root folder with class subfolders
        batch_size: Batch size for dataloaders
        img_size: Target image size
        val_split: Fraction of data for validation
        random_state: Random seed for split
        augment: Enable standard geometric/color augmentations
        min_count: Minimum images per class
        paper_aug: PaperAugmentation instance for paper/lighting effects (optional)

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

    train_transform, val_transform = get_transforms(img_size=img_size, augment=augment, paper_aug=paper_aug)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=random_state
    )

    train_ds = BaybayinDataset(train_paths, train_labels, transform=train_transform)
    val_ds = BaybayinDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, classes
