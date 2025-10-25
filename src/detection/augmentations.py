"""
Detection Augmentation Module
Provides realistic augmentations for Baybayin character detection training.
Includes photometric, lighting, and geometric transformations with bbox handling.
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
import torch


class DetectionAugmentation:
    """
    Augmentation pipeline for object detection training.
    Applies photometric, lighting, and geometric transformations while preserving bbox integrity.
    """
    
    def __init__(
        self,
        # Enable flags
        enable_photometric: bool = True,
        enable_lighting: bool = True,
        enable_geometric: bool = True,
        
        # Photometric parameters
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        blur_kernel_range: Tuple[int, int] = (0, 3),  # 0 means no blur
        noise_std_range: Tuple[float, float] = (0, 5),
        
        # Lighting parameters
        shadow_prob: float = 0.3,
        shadow_intensity_range: Tuple[float, float] = (0.3, 0.7),
        shadow_size_small_prob: float = 0.4,   # Small shadows
        shadow_size_medium_prob: float = 0.4,  # Medium shadows
        shadow_size_large_prob: float = 0.2,   # Large shadows
        # Probability of using a single large cast shadow (e.g. phone shadow)
        shadow_single_prob: float = 0.3,
        overhead_prob: float = 0.2,
        overhead_intensity_range: Tuple[float, float] = (0.1, 0.3),
        spotlight_prob: float = 0.15,
        spotlight_radius_range: Tuple[float, float] = (0.3, 0.6),
        # Spotlight intensity range (multiplier applied to the spotlight mask).
        # Higher values make the spotlight effect stronger/brighter.
        spotlight_intensity_range: Tuple[float, float] = (0.2, 0.5),
        # If True, reduce spotlight strength when the local region is already bright
        spotlight_adapt_to_brightness: bool = False,
        vignette_prob: float = 0.25,
        vignette_intensity_range: Tuple[float, float] = (0.2, 0.5),
        ambient_color_shift_range: Tuple[float, float] = (0, 15),
        
        # Geometric parameters
        rotation_range: Tuple[float, float] = (-5, 5),  # Reduced from ±10 to ±5 to keep symbols visible
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translate_range: Tuple[float, float] = (-0.05, 0.05),
        shear_range: Tuple[float, float] = (-5, 5),
        
        # Probability controls
        photometric_prob: float = 0.8,
        lighting_prob: float = 0.6,
        geometric_prob: float = 0.7,
    ):
        self.enable_photometric = enable_photometric
        self.enable_lighting = enable_lighting
        self.enable_geometric = enable_geometric
        
        # Photometric
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.blur_kernel_range = blur_kernel_range
        self.noise_std_range = noise_std_range
        
        # Lighting
        self.shadow_prob = shadow_prob
        self.shadow_intensity_range = shadow_intensity_range
        self.shadow_size_small_prob = shadow_size_small_prob
        self.shadow_size_medium_prob = shadow_size_medium_prob
        self.shadow_size_large_prob = shadow_size_large_prob
        self.shadow_single_prob = shadow_single_prob
        self.overhead_prob = overhead_prob
        self.overhead_intensity_range = overhead_intensity_range
        self.spotlight_prob = spotlight_prob
        self.spotlight_radius_range = spotlight_radius_range
        self.spotlight_intensity_range = spotlight_intensity_range
        self.spotlight_adapt_to_brightness = spotlight_adapt_to_brightness
        self.vignette_prob = vignette_prob
        self.vignette_intensity_range = vignette_intensity_range
        self.ambient_color_shift_range = ambient_color_shift_range
        
        # Geometric
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.shear_range = shear_range
        
        # Probabilities
        self.photometric_prob = photometric_prob
        self.lighting_prob = lighting_prob
        self.geometric_prob = geometric_prob
    
    def __call__(
        self, 
        image: np.ndarray, 
        bboxes: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations to image and bboxes.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            bboxes: Bounding boxes as numpy array (N, 4) in [x1, y1, x2, y2] format
            labels: Class labels as numpy array (N,)
        
        Returns:
            Augmented image, transformed bboxes, labels (unchanged)
        """
        # Convert to float for processing
        image = image.astype(np.float32)
        bboxes = bboxes.copy()
        
        # Apply augmentations in order
        if self.enable_photometric and random.random() < self.photometric_prob:
            image = self._apply_photometric(image)
        
        if self.enable_lighting and random.random() < self.lighting_prob:
            image = self._apply_lighting(image)
        
        if self.enable_geometric and random.random() < self.geometric_prob:
            image, bboxes = self._apply_geometric(image, bboxes)
        
        # Clip and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image, bboxes, labels
    
    def _apply_photometric(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric transformations."""
        # Brightness
        brightness = random.uniform(*self.brightness_range)
        image = image * brightness
        
        # Contrast
        contrast = random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        # Gamma correction
        gamma = random.uniform(*self.gamma_range)
        image = np.power(image / 255.0, gamma) * 255.0
        
        # Gaussian blur
        kernel_size = random.randint(*self.blur_kernel_range)
        if kernel_size > 0:
            kernel_size = kernel_size * 2 + 1  # Make odd
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Gaussian noise
        noise_std = random.uniform(*self.noise_std_range)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape)
            image = image + noise
        
        return image
    
    def _apply_lighting(self, image: np.ndarray) -> np.ndarray:
        """Apply lighting effects."""
        h, w = image.shape[:2]
        
        # Shadow casting: either a single large cast (phone-like) or multiple small casts
        if random.random() < self.shadow_prob:
            if random.random() < self.shadow_single_prob:
                image = self._add_single_cast_shadow(image)
            else:
                num_shadows = random.randint(1, 3)
                for _ in range(num_shadows):
                    image = self._add_shadow(image)
        
        # Overhead lighting gradient
        if random.random() < self.overhead_prob:
            intensity = random.uniform(*self.overhead_intensity_range)
            direction = random.choice(['top', 'bottom', 'left', 'right'])
            image = self._add_directional_light(image, direction, intensity)
        
        # Spotlight effect
        if random.random() < self.spotlight_prob:
            radius = random.uniform(*self.spotlight_radius_range)
            center_x = random.uniform(0.3, 0.7) * w
            center_y = random.uniform(0.3, 0.7) * h
            image = self._add_spotlight(image, center_x, center_y, radius)
        
        # Vignette
        if random.random() < self.vignette_prob:
            intensity = random.uniform(*self.vignette_intensity_range)
            image = self._add_vignette(image, intensity)
        
        # Ambient color shift
        color_shift = random.uniform(*self.ambient_color_shift_range)
        if color_shift > 0:
            # Shift toward warm or cool
            if random.random() < 0.5:
                # Warm (increase red/yellow)
                image[:, :, 0] += color_shift  # R
                image[:, :, 1] += color_shift * 0.5  # G
            else:
                # Cool (increase blue)
                image[:, :, 2] += color_shift  # B
        
        return image
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add a random shadow cast with size variation."""
        h, w = image.shape[:2]
        intensity = random.uniform(*self.shadow_intensity_range)
        
        # Determine shadow size based on probabilities
        size_choice = random.random()
        total_prob = self.shadow_size_small_prob + self.shadow_size_medium_prob + self.shadow_size_large_prob
        norm_small = self.shadow_size_small_prob / total_prob
        norm_medium = self.shadow_size_medium_prob / total_prob
        
        if size_choice < norm_small:
            # Small shadow (15-35% of image)
            coverage = random.uniform(0.25, 0.45)
        elif size_choice < norm_small + norm_medium:
            # Medium shadow (35-60% of image)
            coverage = random.uniform(0.45, 0.70)
        else:
            # Large shadow (60-85% of image)
            coverage = random.uniform(0.70, 0.95)
        
        # Create random polygon for shadow shape
        num_points = random.randint(3, 6)
        points = []
        
        # Generate points within a constrained region based on coverage
        center_x = random.uniform(0.2, 0.8) * w
        center_y = random.uniform(0.2, 0.8) * h
        radius_x = coverage * w / 2
        radius_y = coverage * h / 2
        
        for _ in range(num_points):
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0.5, 1.0)  # Vary distance from center
            x = int(center_x + r * radius_x * np.cos(angle))
            y = int(center_y + r * radius_y * np.sin(angle))
            # Clamp to image boundaries
            x = max(0, min(w, x))
            y = max(0, min(h, y))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [points], 1.0)
        
        # Apply Gaussian blur to soften edges (blur amount scales with size)
        blur_kernel = int(51 * (coverage + 0.3))  # Larger shadows get softer edges
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        
        # Apply shadow
        shadow_factor = 1.0 - (mask * intensity)
        image = image * shadow_factor[:, :, np.newaxis]
        
        return image

    def _add_single_cast_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add a single large cast shadow (e.g. phone or object cast over the page).

        This creates a large elliptical shadow with soft edges placed toward
        an edge/near-top region to simulate a phone/camera hand casting a shadow.
        """
        h, w = image.shape[:2]
        intensity = random.uniform(*self.shadow_intensity_range)

        # Large coverage for cast shadow (50% - 95% of image)
        coverage = random.uniform(0.5, 0.95)

        # Place the cast shadow toward an edge (phone usually near top/side).
        center_x = int(random.uniform(0.2, 0.8) * w)
        center_y = int(random.uniform(0.0, 0.35) * h)

        radius_x = int(max(1, coverage * w / 2))
        radius_y = int(max(1, coverage * h / 2))

        # Build mask with filled ellipse
        mask = np.zeros((h, w), dtype=np.float32)
        try:
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), int(random.uniform(-30, 30)), 0, 360, 1.0, -1)
        except Exception:
            # Fallback: draw a filled circle if ellipse fails
            cv2.circle(mask, (center_x, center_y), max(radius_x, radius_y), 1.0, -1)

        # Soften edges: blur amount scales with coverage and image size
        blur_kernel = int(min(max(h, w) * 0.5 * coverage, 201))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        # Ensure kernel is at least 3
        blur_kernel = max(3, blur_kernel)
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

        # Apply multiplicative shadow darkening
        shadow_factor = 1.0 - (mask * intensity)
        image = image * shadow_factor[:, :, np.newaxis]

        return image
    
    def _add_directional_light(
        self, 
        image: np.ndarray, 
        direction: str, 
        intensity: float
    ) -> np.ndarray:
        """Add directional lighting gradient."""
        h, w = image.shape[:2]
        
        if direction == 'top':
            gradient = np.linspace(intensity, 0, h)[:, np.newaxis, np.newaxis]
        elif direction == 'bottom':
            gradient = np.linspace(0, intensity, h)[:, np.newaxis, np.newaxis]
        elif direction == 'left':
            gradient = np.linspace(intensity, 0, w)[np.newaxis, :, np.newaxis]
        else:  # right
            gradient = np.linspace(0, intensity, w)[np.newaxis, :, np.newaxis]
        
        gradient = np.broadcast_to(gradient, image.shape)
        image = image + (255 * gradient)
        
        return image
    
    def _add_spotlight(
        self, 
        image: np.ndarray, 
        center_x: float, 
        center_y: float, 
        radius_factor: float
    ) -> np.ndarray:
        """Add spotlight effect."""
        h, w = image.shape[:2]
        
        # Create radial gradient
        y, x = np.ogrid[:h, :w]
        radius = radius_factor * min(h, w)
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create falloff
        spotlight = np.clip(1.0 - (dist / radius), 0, 1)
        spotlight = spotlight[:, :, np.newaxis]
        
        # Brighten center additively (more visible on light backgrounds).
        # Use additive brightness instead of multiplicative so the spotlight
        # shows up even when paper is near-white (avoids saturation hiding effect).
        intensity = random.uniform(*self.spotlight_intensity_range)

        # Optionally adapt intensity to local brightness so very bright
        # backgrounds don't produce an overexposed spot.
        if getattr(self, 'spotlight_adapt_to_brightness', False):
            # Compute mean brightness inside the spotlight mask
            mask_mean = np.sum(image * spotlight) / (np.sum(spotlight) + 1e-9)
            # mask_mean is average per-channel; reduce to 0..255 by averaging channels
            if image.ndim == 3:
                # mask_mean currently sums across channels equally; divide by 3
                local_mean = mask_mean / 3.0
            else:
                local_mean = mask_mean
            # Scale down intensity linearly with local brightness
            adapt_scale = 1.0 - (local_mean / 255.0)
            # Clamp so we still get some visible effect (avoid zeroing)
            adapt_scale = float(np.clip(adapt_scale, 0.15, 1.0))
            intensity = intensity * adapt_scale

        image = image + (spotlight * 255.0 * intensity)

        return image
    
    def _add_vignette(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add vignette effect (darkened corners)."""
        h, w = image.shape[:2]
        
        # Create radial gradient from center
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Normalized distance from center
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
        
        # Create vignette mask
        vignette = 1.0 - (dist * intensity)
        vignette = vignette[:, :, np.newaxis]
        
        image = image * vignette
        
        return image
    
    def _apply_geometric(
        self, 
        image: np.ndarray, 
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric transformations with bbox updates."""
        h, w = image.shape[:2]
        
        # Note: Horizontal flip removed - Baybayin symbols cannot be flipped horizontally
        
        # Build affine transformation matrix
        center = (w / 2, h / 2)
        
        # Start with identity matrix (3x3 homogeneous coordinates)
        M = np.eye(3, dtype=np.float32)
        
        # Rotation around center
        angle = random.uniform(*self.rotation_range)
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Translate to origin, rotate, translate back
        M_rot = np.array([
            [cos_a, -sin_a, center[0] - center[0] * cos_a + center[1] * sin_a],
            [sin_a, cos_a, center[1] - center[0] * sin_a - center[1] * cos_a],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Scale
        scale = random.uniform(*self.scale_range)
        M_scale = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Shear
        shear = random.uniform(*self.shear_range)
        shear_rad = np.deg2rad(shear)
        M_shear = np.array([
            [1, np.tan(shear_rad), 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Translation
        tx = random.uniform(*self.translate_range) * w
        ty = random.uniform(*self.translate_range) * h
        M_trans = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine transformations
        M = M_trans @ M_shear @ M_scale @ M_rot
        
        # Extract 2x3 matrix for cv2.warpAffine
        M_affine = M[:2, :]
        
        # Extract 2x3 matrix for cv2.warpAffine
        M_affine = M[:2, :]
        
        # Apply to image
        image = cv2.warpAffine(
            image, 
            M_affine, 
            (w, h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        # Transform bboxes (use full 3x3 matrix)
        bboxes = self._transform_bboxes(bboxes, M_affine, w, h)
        
        return image, bboxes
    
    def _transform_bboxes(
        self, 
        bboxes: np.ndarray, 
        M: np.ndarray, 
        img_w: int, 
        img_h: int
    ) -> np.ndarray:
        """
        Transform bounding boxes using affine matrix.
        Converts corners, applies transform, computes new bbox.
        """
        if len(bboxes) == 0:
            return bboxes
        
        transformed_bboxes = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Get all 4 corners
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1]
            ], dtype=np.float32).T
            
            # Apply transformation
            transformed_corners = M @ corners
            transformed_corners = transformed_corners.T
            
            # Get new bounding box
            new_x1 = np.min(transformed_corners[:, 0])
            new_y1 = np.min(transformed_corners[:, 1])
            new_x2 = np.max(transformed_corners[:, 0])
            new_y2 = np.max(transformed_corners[:, 1])
            
            # Clip to image bounds
            new_x1 = np.clip(new_x1, 0, img_w)
            new_y1 = np.clip(new_y1, 0, img_h)
            new_x2 = np.clip(new_x2, 0, img_w)
            new_y2 = np.clip(new_y2, 0, img_h)
            
            # Keep bbox if it has valid area
            if new_x2 > new_x1 and new_y2 > new_y1:
                transformed_bboxes.append([new_x1, new_y1, new_x2, new_y2])
        
        return np.array(transformed_bboxes, dtype=np.float32) if transformed_bboxes else np.empty((0, 4), dtype=np.float32)


def create_augmentation_from_args(args) -> Optional[DetectionAugmentation]:
    """
    Create DetectionAugmentation instance from command-line arguments.
    
    Args:
        args: Parsed arguments from argparse
    
    Returns:
        DetectionAugmentation instance if enabled, None otherwise
    """
    if not args.aug_enable:
        return None
    
    return DetectionAugmentation(
        # Enable flags
        enable_photometric=args.aug_photometric,
        enable_lighting=args.aug_lighting,
        enable_geometric=args.aug_geometric,
        
        # Photometric
        brightness_range=(args.aug_brightness_min, args.aug_brightness_max),
        contrast_range=(args.aug_contrast_min, args.aug_contrast_max),
        gamma_range=(args.aug_gamma_min, args.aug_gamma_max),
        blur_kernel_range=(args.aug_blur_min, args.aug_blur_max),
        noise_std_range=(args.aug_noise_min, args.aug_noise_max),
        
        # Lighting
        shadow_prob=args.aug_shadow_prob,
        shadow_intensity_range=(args.aug_shadow_intensity_min, args.aug_shadow_intensity_max),
        shadow_size_small_prob=args.aug_shadow_size_small_prob,
        shadow_size_medium_prob=args.aug_shadow_size_medium_prob,
        shadow_size_large_prob=args.aug_shadow_size_large_prob,
        overhead_prob=args.aug_overhead_prob,
        overhead_intensity_range=(args.aug_overhead_intensity_min, args.aug_overhead_intensity_max),
        spotlight_prob=args.aug_spotlight_prob,
        spotlight_radius_range=(args.aug_spotlight_radius_min, args.aug_spotlight_radius_max),
        vignette_prob=args.aug_vignette_prob,
        vignette_intensity_range=(args.aug_vignette_intensity_min, args.aug_vignette_intensity_max),
        ambient_color_shift_range=(args.aug_ambient_min, args.aug_ambient_max),
        
        # Geometric
        rotation_range=(args.aug_rotation_min, args.aug_rotation_max),
        scale_range=(args.aug_scale_min, args.aug_scale_max),
        translate_range=(args.aug_translate_min, args.aug_translate_max),
        shear_range=(args.aug_shear_min, args.aug_shear_max),
        
        # Probabilities
        photometric_prob=args.aug_photometric_prob,
        lighting_prob=args.aug_lighting_prob,
        geometric_prob=args.aug_geometric_prob,
    )
