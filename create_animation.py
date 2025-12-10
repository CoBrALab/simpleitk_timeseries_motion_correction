#!/usr/bin/env python
import os

import argparse
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import concurrent.futures
from scipy.ndimage import zoom, center_of_mass
from PIL import Image, ImageDraw, ImageFont

def normalize_frame(frame):
    """Normalize frame to 0-255 uint8"""
    f_min = frame.min()
    f_max = frame.max()
    if f_max != f_min:
        frame = (frame - f_min) / (f_max - f_min)
    else:
        frame = np.zeros_like(frame)
    return (frame * 255).astype(np.uint8)

def annotate_frame(image_array, frame_idx):
    """Draw frame number on the bottom right of the image"""
    img_pil = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img_pil)
    
    # Text to draw
    text = f"{frame_idx}"
    
    # Use default font, or try to load one
    # Default font is often very small.
    # Try to use a larger font if available, or just scale up?
    # PIL simplistic font handling...
    # Let's try to load a potentially available font, or fallback to default.
    try:
        # Linux common font path?
        # DejaVuSans.ttf is common.
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except IOError:
        try:
             # Try another one
             font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
             font = ImageFont.load_default()
    
    # Calculate text position
    # textbbox(xy, text, font=None, anchor=None, spacing=4, align='left', direction=None, features=None, language=None, stroke_width=0, embedded_color=False)
    # pillow < 9.2 might use textsize. Checking env? Assuming modern.
    # If old PIL, convert to textsize.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older PIL
        text_w, text_h = draw.textsize(text, font=font)
        
    w, h = img_pil.size
    x = w - text_w - 5 # 5px padding
    y = h - text_h - 5
    
    # Draw text with outline/shadow for visibility?
    # Simple white text
    draw.text((x, y), text, fill=255, font=font)
    
    return np.array(img_pil)

def create_ortho_slice_row(volume, cx, cy, cz, spacing):
    """
    Creates a single row of orthogonal slices (Sagittal, Coronal, Axial)
    centered at (cx, cy, cz), corrected for physical spacing.
    
    volume: 3D numpy array (z, y, x)
    cx, cy, cz: Integers, indices for x, y, z slices.
    spacing: Tuple (sp_z, sp_y, sp_x)
    """
    z_dim, y_dim, x_dim = volume.shape
    sp_z, sp_y, sp_x = spacing
    
    # Target isotropic spacing (min resolution)
    min_sp = min(spacing)

    # Clamp indices just in case
    cx = max(0, min(cx, x_dim - 1))
    cy = max(0, min(cy, y_dim - 1))
    cz = max(0, min(cz, z_dim - 1))

    # 1. Sagittal (X-slice) -> (z, y)
    # volume[:, :, cx]. Flipud to correct orientation (Z up).
    slice_sagittal = np.flipud(volume[:, :, cx])
    # Zoom factors: Z*sp_z / min_sp, Y*sp_y / min_sp
    zoom_sag = (sp_z / min_sp, sp_y / min_sp)
    slice_sagittal = zoom(slice_sagittal, zoom_sag, order=1)

    # 2. Coronal (Y-slice) -> (z, x)
    # volume[:, cy, :]. Flipud to correct orientation (Z up).
    slice_coronal = np.flipud(volume[:, cy, :])
    # Zoom factors: Z*sp_z / min_sp, X*sp_x / min_sp
    zoom_cor = (sp_z / min_sp, sp_x / min_sp)
    slice_coronal = zoom(slice_coronal, zoom_cor, order=1)

    # 3. Axial (Z-slice) -> (y, x)
    # volume[cz, :, :]. Orientation usually OK? Or needs flip?
    # Usually Anterior is up? If Y=0 is top...
    # Let's keep raw for now, consistent with previous.
    slice_axial = volume[cz, :, :]
    # Zoom factors: Y*sp_y / min_sp, X*sp_x / min_sp
    zoom_ax = (sp_y / min_sp, sp_x / min_sp)
    slice_axial = zoom(slice_axial, zoom_ax, order=1)

    # List of slices
    slices = [slice_sagittal, slice_coronal, slice_axial]
    
    # Scale to matching HEIGHT
    # Since we want SAME SCALE, we should PAD to max height, not stretch.
    heights = [s.shape[0] for s in slices]
    max_height = max(heights)

    final_slices = []
    for s in slices:
        h, w = s.shape
        if h < max_height:
            pad_top = (max_height - h) // 2
            pad_bottom = max_height - h - pad_top
            # Pad with zeros (or min value?)
            s_padded = np.pad(s, ((pad_top, pad_bottom), (0, 0)), mode='constant')
            final_slices.append(s_padded)
        else:
            final_slices.append(s)

    return np.hstack(final_slices)

def process_frame(vol, vol2, cx, cy, cz, spacing, scale, frame_idx):
    """Helper function for parallel frame processing"""
    row = create_ortho_slice_row(vol, cx, cy, cz, spacing)
    row_norm = normalize_frame(row)
    
    if vol2 is not None:
        row2 = create_ortho_slice_row(vol2, cx, cy, cz, spacing)
        row2_norm = normalize_frame(row2)
        # Stack vertically
        combined = np.vstack((row_norm, row2_norm))
    else:
        combined = row_norm
        
    # Annotate the bottom of the image
    combined = annotate_frame(combined, frame_idx)
    
    # Scale up
    if scale != 1.0:
        combined = zoom(combined, scale, order=1)
    return combined

def main(input_file, output_file, second_input_file=None, scale=2.0, fps=10):
    print(f"Loading {input_file}...")
    img = sitk.ReadImage(input_file)
    
    # Get array (t, z, y, x) or (z, y, x)
    arr = sitk.GetArrayFromImage(img)
    
    # Get Spacing (x, y, z) -> need (z, y, x)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    print(f"Image Spacing (z, y, x): {spacing_zyx}")
    
    if arr.ndim == 3:
         arr = arr[np.newaxis, ...]
    
    nt, nz, ny, nx = arr.shape
    print(f"Data shape: {arr.shape}")
    
    arr2 = None
    if second_input_file:
        print(f"Loading second input {second_input_file}...")
        img2 = sitk.ReadImage(second_input_file)
        arr2 = sitk.GetArrayFromImage(img2)
        if arr2.ndim == 3:
            arr2 = arr2[np.newaxis, ...]
        
        print(f"Second Data shape: {arr2.shape}")
        if arr2.shape != arr.shape:
             print("WARNING: Shapes do not match exactly. Proceeding with assumption that dimensions are compatible for slicing.")
             # We assume nt is same or we zip?
             # If nt differs, we might crash on indexing if arr2 is shorter.
             # Let's trust user assumption "Assume... same dimensions".
    
    # Calculate Center of Mass on the first frame
    print("Calculating center of mass...")
    # center_of_mass returns (z, y, x) floats
    com_z, com_y, com_x = center_of_mass(arr[0])
    cz, cy, cx = int(round(com_z)), int(round(com_y)), int(round(com_x))
    print(f"Center of Mass indices: x={cx}, y={cy}, z={cz}")
    
    frames = []
    print(f"Generating frames (Parallel using {os.cpu_count() or 1} cores)...")
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Prepare arguments for map
        args_vol = [arr[t] for t in range(nt)]
        args_idx = list(range(nt))
        
        if arr2 is not None:
            # Handle case where len(arr2) != nt? User said assume dimensions same.
            args_vol2 = [arr2[t] for t in range(nt)]
        else:
            args_vol2 = [None] * nt
            
        # map preserves order
        results = executor.map(process_frame, args_vol, args_vol2, [cx]*nt, [cy]*nt, [cz]*nt, [spacing_zyx]*nt, [scale]*nt, args_idx)
        
        frames = list(results)
        
    print(f"Saving to {output_file}...")
    # imageio v3 imwrite supports loop and duration/fps for webp
    # fps is supported by pillow plugin for webp
    iio.imwrite(output_file, frames, duration=1000/fps, loop=0)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create animation from NIfTI timeseries")
    parser.add_argument("input", help="Input NIfTI file")
    parser.add_argument("output", help="Output file (e.g. .webp, .gif, .mp4)")
    parser.add_argument("--second-input", help="Optional second input NIfTI file (same dimensions) to display as second row")
    parser.add_argument("--scale", type=float, default=2.0, help="Scale factor for upsampling (default: 2.0)")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.second_input, args.scale, args.fps)
