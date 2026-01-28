#!/usr/bin/env python
import os
import csv
import concurrent.futures
from tqdm import tqdm
import SimpleITK as sitk


def read_transforms_from_csv(csv_file):
    """
    Reads transforms from a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.

    Returns:
        list: List of sitk.Euler3DTransform objects.
    """
    transforms = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "NaN" in row.values():
                transforms.append(sitk.Euler3DTransform())
                continue

            try:
                transform = sitk.Euler3DTransform()

                # Set parameters (Euler angles + Translation)
                # Parameters order in GetParameters for Euler3D is (angleX, angleY, angleZ, transX, transY, transZ)
                params = [
                    float(row["EulerX"]),
                    float(row["EulerY"]),
                    float(row["EulerZ"]),
                    float(row["TransX"]),
                    float(row["TransY"]),
                    float(row["TransZ"]),
                ]
                transform.SetParameters(params)

                # Set Fixed Center
                center = [
                    float(row["RotCenterX"]),
                    float(row["RotCenterY"]),
                    float(row["RotCenterZ"]),
                ]
                transform.SetCenter(center)

                transforms.append(transform)
            except ValueError:
                # Fallback for malformed rows
                transforms.append(sitk.Euler3DTransform())

    return transforms


def resample_volume(
    volume,
    reference,
    transform,
    interpolation=sitk.sitkBSpline5,
    clip_negative=True,
    extrapolator=True,
):
    """
    Resamples a single 3D volume to the reference geometry.
    """
    resampled = sitk.Resample(
        volume,
        reference,
        transform,
        interpolation,
        0.0,  # Default pixel value
        volume.GetPixelID(),
        useNearestNeighborExtrapolator=extrapolator,
    )
    if clip_negative:
        resampled = resampled * sitk.Cast(resampled > 0, resampled.GetPixelID())
    return resampled


def framewise_resample_volume(
    input_image,
    reference_image,
    transforms,
    interpolation=sitk.sitkBSpline5,
    clip_negative=True,
    extrapolator=True,
    max_workers=os.cpu_count(),
):
    num_volumes = input_image.GetSize()[3]
    # Extract 3D volumes from input to process in parallel
    input_volumes = []
    size_4d = input_image.GetSize()
    for i in range(num_volumes):
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size_4d[0], size_4d[1], size_4d[2], 0])
        extractor.SetIndex([0, 0, 0, i])
        input_volumes.append(extractor.Execute(input_image))

    resampled_volumes = [None] * num_volumes

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=num_volumes) as pbar:
            futures = {}
            for i in range(num_volumes):
                future = executor.submit(
                    resample_volume,
                    volume=input_volumes[i],
                    reference=reference_image,
                    transform=transforms[i],
                    interpolation=interpolation,
                    clip_negative=clip_negative,
                    extrapolator=extrapolator,
                )
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    resampled_volumes[i] = future.result()
                except Exception as e:
                    print(f"Error processing volume {i}: {e}")
                    # Might want to abort or insert blank?
                    # For now re-raise to fail fast
                    raise e
                pbar.update(1)

    print("Joining series...")
    return sitk.JoinSeries(resampled_volumes)


def main(args):
    print(f"Reading input: {args.input_file}")
    input_image = sitk.ReadImage(args.input_file)
    # Check if 4D
    if input_image.GetDimension() != 4:
        raise ValueError("Input image must be 4D.")

    num_volumes = input_image.GetSize()[3]

    print(f"Reading reference: {args.reference_file}")
    reference_image = sitk.ReadImage(args.reference_file)
    # Handle 4D reference by taking first volume
    if reference_image.GetDimension() == 4:
        print("Reference image is 4D, using the first volume as geometry target.")
        extractor = sitk.ExtractImageFilter()
        size = reference_image.GetSize()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, 0])
        reference_image = extractor.Execute(reference_image)

    print(f"Reading transforms from: {args.csv_file}")
    transforms = read_transforms_from_csv(args.csv_file)

    if len(transforms) != num_volumes:
        raise ValueError(
            f"Number of transforms ({len(transforms)}) does not match input volumes ({num_volumes})."
        )

    print("Resampling volumes...")
    output_image = framewise_resample_volume(
        input_image,
        reference_image,
        transforms,
        interpolation=sitk.sitkBSpline5,
        clip_negative=True,
        extrapolator=True,
    )

    print(f"Writing output to: {args.output_file}")
    sitk.WriteImage(output_image, args.output_file)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply transforms from CSV to a time series."
    )
    parser.add_argument("input_file", help="Input 4D fMRI file (NIfTI format)")
    parser.add_argument("csv_file", help="CSV file containing transforms")
    parser.add_argument("reference_file", help="Reference image (NIfTI format)")
    parser.add_argument("output_file", help="Output motion corrected file")

    args = parser.parse_args()
    main(args)
