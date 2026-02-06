"""SimpleITK Timeseries Motion Correction

A Python package for fMRI motion correction using SimpleITK.
"""

# Motion correction functions
from .motion import (
    framewise_register_pair,
    register_pair,
    register_slice_pair,
    resample_slice_pair,
    write_transforms_to_csv,
    isotropic_upsample_and_pad,
    voxelwise_mean,
    voxelwise_std,
    make_mask,
)

# Transform application functions
from .apply_transforms import (
    read_transforms_from_csv,
    resample_volume,
    framewise_resample_volume,
)

# Framewise displacement calculation
from .framewise_displacement import (
    calculate_framewise_displacement,
)

# Metrics computation
from .plot_metrics import (
    compute_metrics,
)

# Animation creation
from .create_animation import (
    create_ortho_slice_row,
    normalize_frame,
)

# Estimate registration parameters
from .make_pyramid import (
    make_pyramid,
)

__all__ = [
    # Motion correction
    "framewise_register_pair",
    "register_pair",
    "register_slice_pair",
    "resample_slice_pair",
    "write_transforms_to_csv",
    "isotropic_upsample_and_pad",
    "voxelwise_mean",
    "voxelwise_std",
    "make_mask",
    # Transform application
    "read_transforms_from_csv",
    "resample_volume",
    "framewise_resample_volume",
    # Framewise displacement
    "calculate_framewise_displacement",
    # Metrics
    "compute_metrics",
    # Animation
    "create_ortho_slice_row",
    "normalize_frame",
    # Estimate registration parameters
    "make_pyramid",    
]
