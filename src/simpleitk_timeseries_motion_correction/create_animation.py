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

    font = ImageFont.load_default()

    # Calculate text position
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older PIL
        text_w, text_h = draw.textsize(text, font=font)

    w, h = img_pil.size
    x = w - text_w - 5  # 5px padding
    y = h - text_h - 5

    # Simple white text
    draw.text((x, y), text, fill=255, font=font)

    return np.array(img_pil)


def draw_label(image_array, text):
    """Draw text on the top left of the image"""
    img_pil = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.load_default(8)

    x = 5
    y = 5

    # Draw text
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
    slice_sagittal = np.flipud(volume[:, :, cx])
    # Zoom factors: Z*sp_z / min_sp, Y*sp_y / min_sp
    zoom_sag = (sp_z / min_sp, sp_y / min_sp)
    slice_sagittal = zoom(slice_sagittal, zoom_sag, order=1)

    # 2. Coronal (Y-slice) -> (z, x)
    slice_coronal = np.flipud(volume[:, cy, :])
    # Zoom factors: Z*sp_z / min_sp, X*sp_x / min_sp
    zoom_cor = (sp_z / min_sp, sp_x / min_sp)
    slice_coronal = zoom(slice_coronal, zoom_cor, order=1)

    # 3. Axial (Z-slice) -> (y, x)
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
            s_padded = np.pad(s, ((pad_top, pad_bottom), (0, 0)), mode="constant")
            final_slices.append(s_padded)
        else:
            final_slices.append(s)

    return np.hstack(final_slices)


def process_frame(
    vol, additional_vols, cx, cy, cz, spacing, scale, frame_idx, labels=None
):
    """Helper function for parallel frame processing"""
    row = create_ortho_slice_row(vol, cx, cy, cz, spacing)
    row_norm = normalize_frame(row)

    if labels and len(labels) > 0:
        row_norm = draw_label(row_norm, labels[0])

    combined = row_norm

    if additional_vols:
        for i, vol_add in enumerate(additional_vols):
            if vol_add is not None:
                row_add = create_ortho_slice_row(vol_add, cx, cy, cz, spacing)
                row_add_norm = normalize_frame(row_add)

                if labels and len(labels) > i + 1:
                    row_add_norm = draw_label(row_add_norm, labels[i + 1])

                combined = np.vstack((combined, row_add_norm))

    # Annotate the bottom of the image
    combined = annotate_frame(combined, frame_idx)

    # Scale up
    if scale != 1.0:
        combined = zoom(combined, scale, order=1)
    return combined


def main(input_file, output_file, additional_input_files=None, scale=2.0, fps=10):
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

    additional_arrs = []
    if additional_input_files:
        for f in additional_input_files:
            print(f"Loading additional input {f}...")
            img_add = sitk.ReadImage(f)
            arr_add = sitk.GetArrayFromImage(img_add)
            if arr_add.ndim == 3:
                arr_add = arr_add[np.newaxis, ...]
            print(f"Additional Data shape: {arr_add.shape}")
            if arr_add.shape != arr.shape:
                print(
                    f"WARNING: Shape of {f} does not match exactly. Proceeding with assumption that dimensions are compatible for slicing."
                )
            additional_arrs.append(arr_add)

    labels = [os.path.basename(input_file)]
    if additional_input_files:
        labels.extend([os.path.basename(f) for f in additional_input_files])

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

        if additional_arrs:
            # Transpose list of arrays to list of timepoints containing list of arrays
            # additional_arrs is [N_files, T, Z, Y, X] (conceptually, though they are list of numpy arrays)
            # We want args_additional_vol to be length T, where each element is [vol_file1, vol_file2, ...]

            # Assuming all have same T as arr
            args_additional_vols = []
            for t in range(nt):
                vols_at_t = []
                for arr_idx, arr_add in enumerate(additional_arrs):
                    if t < arr_add.shape[0]:
                        vols_at_t.append(arr_add[t])
                    else:
                        vols_at_t.append(None)  # Or handle error
                args_additional_vols.append(vols_at_t)
        else:
            args_additional_vols = [[] for _ in range(nt)]

        # map preserves order

        results = executor.map(
            process_frame,
            args_vol,
            args_additional_vols,
            [cx] * nt,
            [cy] * nt,
            [cz] * nt,
            [spacing_zyx] * nt,
            [scale] * nt,
            args_idx,
            [labels] * nt,
        )

        frames = list(results)

    print(f"Saving to {output_file}...")
    # imageio v3 imwrite supports loop and duration/fps for webp
    # fps is supported by pillow plugin for webp
    iio.imwrite(output_file, frames, duration=1000 / fps, loop=0)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create animation from NIfTI timeseries"
    )
    parser.add_argument("input", help="Input NIfTI file")
    parser.add_argument("output", help="Output file (e.g. .webp, .gif, .mp4)")
    parser.add_argument(
        "--additional-row",
        action="append",
        help="Optional additional input NIfTI file(s) (same dimensions) to display as additional rows",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Scale factor for upsampling (default: 2.0)",
    )
    parser.add_argument("--fps", type=float, default=10, help="Frames per second")

    args = parser.parse_args()

    main(args.input, args.output, args.additional_row, args.scale, args.fps)
