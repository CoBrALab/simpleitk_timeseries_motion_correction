#!/usr/bin/env python

import argparse
import SimpleITK as sitk
import os
import sys
import numpy as np
from tqdm import tqdm
from apply_transforms import read_transforms_from_csv

def calculate_framewise_displacement(mask_file, csv_file, output_csv=None, verbose=False):
    if verbose:
        print(f"Reading mask: {mask_file}")
    mask = sitk.ReadImage(mask_file)
    
    # Get all points inside the mask
    if verbose:
        print("Extracting ROI points...")
    mask_arr = sitk.GetArrayViewFromImage(mask)
    # numpy is (z,y,x), sitk needs (x,y,z)
    z_idxs, y_idxs, x_idxs = np.where(mask_arr > 0)
    
    points = []
    for z, y, x in zip(z_idxs, y_idxs, x_idxs):
        pt = mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        points.append(pt)
    
    if len(points) == 0:
        print("No points found in mask.")
        return []

    if verbose:
        print(f"ROI contains {len(points)} voxels.")
    
    if verbose:
        print(f"Reading transforms: {csv_file}")
    transforms = read_transforms_from_csv(csv_file)
    
    if len(transforms) < 2:
        print("Not enough transforms.")
        return []
    
    results = []
    if verbose:
        print("Calculating Framewise Displacement (Point Tracking)...")
    

    # Initialize P_prev (t=0)
    # We use the direct transform here because transforming points uses the opposite direction as resampling images
    t0 = transforms[0]
    prev_points = np.array([t0.TransformPoint(p) for p in points])
    
    # First timepoint (t=0) has 0 displacement by definition
    results.append({
        "timepoint": 0,
        "mean_fd": 0.0,
        "max_fd": 0.0
    })

    # Loop over each index looking forwards
    for i in tqdm(range(len(transforms) - 1)):
        # t = i+1
        t_next = transforms[i+1]
        
        # We use the direct transform here because transforming points uses the opposite direction as resampling images
        curr_points = np.array([t_next.TransformPoint(p) for p in points])
        
        # Calculate distances
        distances = np.linalg.norm(curr_points - prev_points, axis=1)
        
        mean_fd = np.mean(distances)
        max_fd = np.max(distances)
        
        results.append({
            "timepoint": i + 1,
            "mean_fd": mean_fd,
            "max_fd": max_fd
        })
        
        prev_points = curr_points

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Timepoint", "MeanFD", "MaxFD"])
            for res in results:
                writer.writerow([
                    res['timepoint'],
                    res['mean_fd'],
                    res['max_fd']
                ])
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate Framewise Displacement from moco.csv and mask using point tracking.")
    parser.add_argument("mask_file", help="Binary mask file in mean space (NIfTI)")
    parser.add_argument("csv_file", help="moco.csv file containing transforms")
    parser.add_argument("--output_csv", help="Optional output CSV for FD values", default=None)
    parser.add_argument("--verbose", help="Optional verbose output", action="store_true")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mask_file):
        print(f"Error: Mask file not found: {args.mask_file}")
        sys.exit(1)
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    calculate_framewise_displacement(args.mask_file, args.csv_file, args.output_csv, args.verbose)

if __name__ == "__main__":
    main()
