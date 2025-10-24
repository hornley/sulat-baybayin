"""
YAML Configuration Manager for Training and Data Generation Scripts

Provides utilities for:
- Generating YAML config templates with defaults and documentation
- Loading and validating YAML configs
- Merging YAML configs with CLI arguments (CLI takes priority)
- Interactive user editing workflow
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def generate_yaml_template(output_path: str, config_dict: Dict[str, Any], header_comment: str = "Configuration File"):
    """Generate a YAML config file with documentation and default values.
    
    Args:
        output_path: Path to write YAML file
        config_dict: Dictionary of config parameters with nested structure
        header_comment: Header comment for the YAML file
    """
    # Create parent directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Define logical groupings for different script types
    # Detect which script by checking for characteristic parameters
    groups = None
    
    # Check if this is generate_synthetic_sentences.py
    if 'paper_type' in config_dict and 'symbol_height_frac' in config_dict:
        groups = [
            ('OUTPUT & GENERATION', ['count', 'out_dir', 'ann', 'append', 'min_symbols', 'max_symbols']),
            ('SYMBOL NORMALIZATION', ['symbol_height_frac', 'bg_thresh_pct', 'crop_pad', 'mask_smooth_radius', 'use_cache', 'cache_dir']),
            ('SHADOW & EROSION', ['erode_shadow', 'erode_shadow_min_thickness', 'erode_shadow_prob', 'erode_glyph', 'erode_glyph_min_thickness', 'erode_glyph_prob']),
            ('INK APPEARANCE', ['ink_color', 'ink_darken_min', 'ink_darken_max', 'ink_alpha_gain', 'ink_alpha_gamma']),
            ('THIN STROKE HELPERS', ['thin_stroke_thresh', 'thin_alpha_gain', 'thin_alpha_gamma', 'thin_darken_boost', 'thin_alpha_floor']),
            ('PAPER TYPE & TEXTURE', ['paper_type', 'paper_type_mix', 'paper_texture', 'paper_strength', 'paper_yellow_strength']),
            ('RULED LINES', ['paper_lines_prob', 'line_spacing', 'line_opacity', 'line_thickness', 'line_jitter', 'line_color']),
            ('DOTTED PAPER', ['dot_size', 'dot_opacity', 'dot_spacing', 'dot_foreground']),
            ('CRUMPLED TEXTURE', ['crumple_strength', 'crumple_mesh_overlap']),
            ('LIGHTING', ['lighting', 'brightness_jitter', 'contrast_jitter', 'shadow_intensity']),
        ]
    
    # Check if this is classification training
    elif 'augment' in config_dict and 'aug_paper_prob' in config_dict:
        groups = [
            ('DATA & OUTPUT', ['data', 'out', 'resume', 'resume_optimizer', 'device']),
            ('TRAINING HYPERPARAMETERS', ['epochs', 'batch', 'lr', 'weight_decay', 'lr_backbone', 'lr_head', 'schedule', 'patience']),
            ('DATALOADER SETTINGS', ['num_workers', 'pin_memory', 'prefetch_factor']),
            ('MIXED PRECISION', ['amp']),
            ('MODEL SETTINGS', ['img_size', 'freeze_backbone', 'min_count', 'weights']),
            ('STANDARD AUGMENTATION', ['augment']),
            ('PAPER TEXTURE AUGMENTATION', ['aug_paper_prob', 'aug_paper_type_probs', 'aug_paper_texture_probs', 'aug_paper_strength_min', 'aug_paper_strength_max', 'aug_paper_yellow_strength_min', 'aug_paper_yellow_strength_max', 'aug_crumple_strength_min', 'aug_crumple_strength_max', 'aug_crumple_mesh_overlap']),
            ('PAPER LINES AUGMENTATION', ['aug_lines_prob', 'aug_line_spacing_min', 'aug_line_spacing_max', 'aug_line_opacity_min', 'aug_line_opacity_max', 'aug_line_thickness_min', 'aug_line_thickness_max', 'aug_line_jitter_min', 'aug_line_jitter_max', 'aug_line_color']),
            ('LIGHTING AUGMENTATION', ['aug_lighting_prob', 'aug_lighting_modes', 'aug_brightness_jitter', 'aug_contrast_jitter', 'aug_shadow_intensity_min', 'aug_shadow_intensity_max']),
            ('DOTTED PAPER OPTIONS', ['aug_dot_size', 'aug_dot_opacity', 'aug_dot_spacing']),
            ('EARLY STOPPING', ['early_stop']),
        ]
    
    # Check if this is detection training
    elif 'freeze_backbone' in config_dict and 'momentum' in config_dict:
        groups = [
            ('DATA & OUTPUT', ['data', 'ann', 'out', 'resume', 'resume_optimizer', 'device']),
            ('TRAINING HYPERPARAMETERS', ['epochs', 'batch', 'lr', 'weight_decay', 'momentum', 'lr_backbone', 'lr_head']),
            ('DATALOADER SETTINGS', ['num_workers', 'pin_memory', 'prefetch_factor']),
            ('MIXED PRECISION', ['amp']),
            ('DETECTION SPECIFIC', ['freeze_backbone', 'save_last', 'no_batch_eval']),
            ('LEARNING RATE SCHEDULE', ['schedule', 'lr_step', 'lr_gamma']),
            ('VALIDATION', ['val_ann']),
            ('REAL/SYNTHETIC MIXING', ['real_data', 'real_ann', 'real_weight', 'mix_strategy']),
            ('EARLY STOPPING', ['early_stop_patience', 'early_stop_min_delta', 'early_stop_monitor', 'early_stop']),
            ('GOOGLE DRIVE BACKUP', ['gdrive_backup']),
        ]
    
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"# {header_comment}\n")
        f.write(f"# Generated automatically - edit as needed\n")
        f.write(f"# \n")
        f.write(f"# CLI arguments override values in this file\n")
        f.write(f"# Use --no-wait to skip confirmation prompt\n")
        f.write(f"# Use --regen-args to regenerate this file from CLI args\n\n")
        
        if groups:
            # Write in grouped format with section headers
            written_keys = set()
            
            for section_name, keys in groups:
                # Check if any keys in this section exist in config_dict
                section_keys = [k for k in keys if k in config_dict]
                if not section_keys:
                    continue
                
                # Write section header
                f.write(f"# === {section_name} ===\n")
                
                # Write each key in this section
                for key in section_keys:
                    value = config_dict[key]
                    # Format the value properly
                    if isinstance(value, str):
                        # Quote strings that contain special characters
                        if ',' in str(value) or ':' in str(value):
                            f.write(f'{key}: "{value}"\n')
                        else:
                            f.write(f'{key}: {value}\n')
                    elif value is None:
                        f.write(f'{key}: null\n')
                    elif isinstance(value, bool):
                        f.write(f'{key}: {str(value).lower()}\n')
                    else:
                        f.write(f'{key}: {value}\n')
                    written_keys.add(key)
                
                f.write('\n')  # Blank line between sections
            
            # Write any remaining keys that weren't in defined groups
            remaining_keys = [k for k in config_dict.keys() if k not in written_keys]
            if remaining_keys:
                f.write(f"# === OTHER SETTINGS ===\n")
                for key in remaining_keys:
                    value = config_dict[key]
                    if isinstance(value, str):
                        if ',' in str(value) or ':' in str(value):
                            f.write(f'{key}: "{value}"\n')
                        else:
                            f.write(f'{key}: {value}\n')
                    elif value is None:
                        f.write(f'{key}: null\n')
                    elif isinstance(value, bool):
                        f.write(f'{key}: {str(value).lower()}\n')
                    else:
                        f.write(f'{key}: {value}\n')
        else:
            # Fallback: use default YAML dump if no grouping detected
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✓ Generated config template: {output_path}")



def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as dictionary.
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        Dictionary of config parameters
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    print(f"✓ Loaded config from: {yaml_path}")
    return config


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary into dot-notation keys.
    
    Example:
        {'paper': {'type': 'white'}} -> {'paper.type': 'white'}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dot-notation keys into nested dictionary.
    
    Example:
        {'paper.type': 'white'} -> {'paper': {'type': 'white'}}
    """
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def merge_configs(yaml_config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Merge YAML config with CLI arguments. CLI arguments take priority.
    
    Args:
        yaml_config: Dictionary from YAML file (can be nested)
        cli_args: Dictionary with CLI arguments (usually from vars(args))
        
    Returns:
        Merged config dictionary with CLI arg names as keys
        
    Example:
        >>> yaml_config = {'count': 100, 'out_dir': 'yaml_out'}
        >>> cli_args = {'count': 200, 'out_dir': 'yaml_out', 'new_arg': 'value'}
        >>> merge_configs(yaml_config, cli_args)
        {'count': 100, 'out_dir': 'yaml_out', 'new_arg': 'value'}
        # 'count' comes from YAML since CLI has same value (no override)
        # 'new_arg' added from CLI
    """
    # Flatten YAML config to handle nested structures
    flat_yaml = flatten_dict(yaml_config) if any(isinstance(v, dict) for v in yaml_config.values()) else yaml_config
    
    # Start with YAML values
    merged = dict(flat_yaml)
    
    # Add/override with CLI args
    for key, cli_value in cli_args.items():
        # Skip if key is in CLI args (add it)
        if key not in merged:
            merged[key] = cli_value
        # If key exists in both, CLI overrides YAML only if different
        elif cli_value != merged.get(key):
            merged[key] = cli_value
    
    return merged


def wait_for_user_edit(config_path: str, timeout_seconds: Optional[int] = None):
    """Wait for user to edit config file and confirm.
    
    Args:
        config_path: Path to config file user should edit
        timeout_seconds: Optional timeout in seconds (None = wait indefinitely)
    """
    print(f"\n{'='*70}")
    print(f"Config file generated: {config_path}")
    print(f"{'='*70}")
    print(f"\nPlease review and edit the configuration file if needed.")
    print(f"Press Enter to continue with the current config, or Ctrl+C to cancel...")
    print(f"{'='*70}\n")
    
    try:
        if timeout_seconds:
            # Note: input() doesn't support timeout in standard Python
            # For production, could use signal.alarm or threading
            input()
        else:
            input()
        print("✓ Continuing with configuration...\n")
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user.")
        sys.exit(0)


def validate_param_range(value: Any, param_name: str, min_val: Optional[float] = None, 
                         max_val: Optional[float] = None, choices: Optional[list] = None) -> bool:
    """Validate parameter is within acceptable range or choices.
    
    Args:
        value: Parameter value to validate
        param_name: Name of parameter (for error messages)
        min_val: Minimum acceptable value (None = no minimum)
        max_val: Maximum acceptable value (None = no maximum)
        choices: List of acceptable values (None = any value OK)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return True
    
    if choices is not None:
        if value not in choices:
            raise ValueError(f"{param_name} must be one of {choices}, got: {value}")
    
    if min_val is not None and isinstance(value, (int, float)):
        if value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got: {value}")
    
    if max_val is not None and isinstance(value, (int, float)):
        if value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got: {value}")
    
    return True


def validate_probability(value: float, param_name: str) -> bool:
    """Validate probability parameter is in [0, 1] range."""
    return validate_param_range(value, param_name, min_val=0.0, max_val=1.0)


def validate_config(config: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]) -> bool:
    """Validate entire config against rules.
    
    Args:
        config: Flat config dictionary
        validation_rules: Dict mapping param names to validation rules
            Example: {'count': {'min': 1, 'max': 100000}, 
                     'paper_type': {'choices': ['white', 'yellow-paper', 'dotted']}}
                     
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If any validation fails
    """
    for param, rules in validation_rules.items():
        if param in config:
            validate_param_range(
                config[param],
                param,
                min_val=rules.get('min'),
                max_val=rules.get('max'),
                choices=rules.get('choices')
            )
    
    return True


def print_config_summary(config: Dict[str, Any], title: str = "Configuration Summary"):
    """Print formatted summary of configuration."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if isinstance(config, dict):
        # If nested, flatten for display
        if any(isinstance(v, dict) for v in config.values()):
            flat = flatten_dict(config)
            for key, value in flat.items():
                print(f"  {key}: {value}")
        else:
            for key, value in config.items():
                print(f"  {key}: {value}")
    
    print(f"{'='*70}\n")
