#!/usr/bin/env python

import SimpleITK as sitk
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm


def compute_metrics(fixed_image, moving_image):
    """
    Computes similarity metrics between two 3D images.
    Returns a dictionary of metrics.
    """
    # Ensure images are float32 for metric calculation
    fixed = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving = sitk.Cast(moving_image, sitk.sitkFloat32)

    metrics = {}

    # Image Registration Method to evaluate metrics
    # We create a new registration method for each metric to ensure clean state

    # Mattes Mutual Information
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.NONE)
    # We need an initial transform to evaluate. Identity is fine.
    tx = sitk.Transform(fixed.GetDimension(), sitk.sitkIdentity)
    R.SetInitialTransform(tx)
    metrics["Mattes Mutual Information"] = R.MetricEvaluate(fixed, moving)

    # Correlation
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInitialTransform(tx)
    metrics["Correlation"] = R.MetricEvaluate(fixed, moving)

    # Mean Squares
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInitialTransform(tx)
    metrics["Mean Squares"] = R.MetricEvaluate(fixed, moving)

    # Joint Histogram Mutual Information
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInitialTransform(tx)
    metrics["Joint Histogram MI"] = R.MetricEvaluate(fixed, moving)

    # Cosine Similarity
    # Flatten images to 1D arrays
    fixed_arr = sitk.GetArrayFromImage(fixed).flatten()
    moving_arr = sitk.GetArrayFromImage(moving).flatten()
    cosine_dist = cosine(fixed_arr, moving_arr)
    metrics["Cosine Distance"] = cosine_dist

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot similarity metrics for 4D timeseries."
    )
    parser.add_argument(
        "--compare",
        required=True,
        action="append",
        help="Pair of images: moving_4d,fixed_3d",
    )
    parser.add_argument("--output", required=True, help="Path to save the output plot")

    args = parser.parse_args()

    # Store results for all pairs
    # Structure: {label: {metric_name: [values]}}
    all_results = {}

    # Common timepoints (assumes all 4D images have same number of timepoints)
    timepoints = None

    for pair_str in args.compare:
        parts = pair_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid compare argument '{pair_str}'. Must be 'moving_image,fixed_image'"
            )

        moving_path, fixed_path = parts[0].strip(), parts[1].strip()

        print(f"\nProcessing pair: Moving='{moving_path}', Fixed='{fixed_path}'")

        fixed_image = sitk.ReadImage(fixed_path)
        moving_image_4d = sitk.ReadImage(moving_path)

        # Verify dimensions
        if fixed_image.GetDimension() != 3:
            raise ValueError(f"Fixed image {fixed_path} must be 3D")
        if moving_image_4d.GetDimension() != 4:
            raise ValueError(f"Moving image {moving_path} must be 4D")

        size_4d = moving_image_4d.GetSize()
        num_timepoints = size_4d[3]
        if timepoints is None:
            timepoints = range(num_timepoints)
        elif len(timepoints) != num_timepoints:
            print(
                f"Warning: {moving_path} has {num_timepoints} timepoints, expected {len(timepoints)}. Plotting may be mismatched."
            )

        print(f"Number of timepoints: {num_timepoints}")

        # Initialize lists to store metrics
        metric_history = {
            "Mattes Mutual Information": [],
            "Correlation": [],
            "Mean Squares": [],
            "Cosine Distance": [],
            "Joint Histogram MI": [],
        }

        # Extract slice filter
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize([size_4d[0], size_4d[1], size_4d[2], 0])

        for t in tqdm(
            range(num_timepoints), desc=f"Processing {moving_path.split('/')[-1]}"
        ):
            extract_filter.SetIndex([0, 0, 0, t])
            moving_frame = extract_filter.Execute(moving_image_4d)

            current_metrics = compute_metrics(fixed_image, moving_frame)

            for key, value in current_metrics.items():
                metric_history[key].append(value)

        label = f"{moving_path.split('/')[-1]} vs {fixed_path.split('/')[-1]}"
        all_results[label] = metric_history

    print("\nComputing completed. Plotting results...")

    # Plotting
    # Plotting
    # 3 rows, 2 columns grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    axes_flat = axes.flatten()

    metrics_to_plot = [
        "Mattes Mutual Information",
        "Correlation",
        "Mean Squares",
        "Cosine Distance",
        "Joint Histogram MI",
    ]

    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes_flat[i]
        for label, history in all_results.items():
            values = history[metric_name]
            ax.plot(
                range(len(values)),
                values,
                marker="o",
                linestyle="-",
                markersize=2,
                label=label,
            )

        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} over Time")
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for j in range(len(metrics_to_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Set x-label for bottom subplots that are visible
    # Simple way with sharex=True: The bottom-most axis in each column column needs label (if visible).
    # 5 plots: indices 0,1,2,3,4.
    # col 0: 0, 2, 4. 4 is bottom.
    # col 1: 1, 3. 3 is bottom visible.

    axes_flat[4].set_xlabel("Timepoint")  # Last plot in col 0
    axes_flat[3].set_xlabel("Timepoint")  # Last plot in col 1

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
