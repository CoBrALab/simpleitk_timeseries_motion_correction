#!/usr/bin/env python

import os
import SimpleITK as sitk
import numpy as np
import concurrent.futures
from tqdm import tqdm
import csv
from apply_transforms import resample_volume

def write_transforms_to_csv(transforms, output_file):
    """
    Writes the parameters of a list of SimpleITK transforms to a CSV file.

    Parameters:
        transforms (list): List of sitk.Transform objects.
        output_file (str): Path to the output CSV file.
    """
    if not transforms:
        return

    # Assuming all transforms have the same number of parameters
    num_params = len(transforms[0].GetParameters())
    # header = [f"param_{i}" for i in range(num_params)] + ['CenterX', 'CenterY', 'CenterZ']
    header = [
        "EulerX",
        "EulerY",
        "EulerZ",
        "TransX",
        "TransY",
        "TransZ",
        "RotCenterX",
        "RotCenterY",
        "RotCenterZ",
    ]
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for transform in transforms:
            if transform:
                writer.writerow(transform.GetParameters() + transform.GetCenter())
            else:
                # Handle None transforms if any (though logic suggests they are filled)
                writer.writerow(["NaN"] * num_params)


def command_iteration(method):
    """Callback invoked when the optimization process is performing an iteration."""
    print(
        f" {method.GetCurrentLevel()} "
        f"{method.GetOptimizerIteration():3} "
        + f"{method.GetOptimizerLearningRate()} "
        + f"= {method.GetMetricValue():.10f} "
        + f": {method.GetOptimizerPosition()}"
    )


def command_iteration2(method):
    """Callback invoked when the optimization process is performing an iteration."""
    print(f"[{method.GetOptimizerIteration()}]", end="", flush=True)


def make_mask(image):
    otsu = sitk.OtsuMultipleThresholds(image, 4, valleyEmphasis=True)
    otsu[otsu > 0.5] = 1
    otsu = sitk.BinaryErode(otsu, [1, 1, 1])
    components = sitk.ConnectedComponent(otsu)
    mask = sitk.RelabelComponent(components) == 1
    mask = sitk.BinaryDilate(mask, [2, 2, 2])
    mask = sitk.BinaryFillhole(mask)
    return mask


def isotropic_upsample_and_pad(image, interpolation=sitk.sitkBSpline5, clip_negative=True):
    """
    Resample the image to isotropic spacing using the smallest existing spacing.

    Parameters:
        image (sitk.Image): Input SimpleITK image.
        interpolation: SimpleITK interpolation method
                       (e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor, sitk.sitkBSpline).

    Returns:
        sitk.Image: Isotropically upsampled image.
    """
    original_spacing = image.GetSpacing()
    min_spacing = min(original_spacing)
    if np.allclose(original_spacing,image.GetDimension()*(min_spacing,)):
        # the image is already isotropic
        resampled_image = image
    else:
        # Compute new size to maintain physical extent
        original_size = image.GetSize()
        new_size = [
            int(round(original_size[i] * original_spacing[i] / min_spacing))
            for i in range(len(original_spacing))
        ]

        # Resample
        resampled_image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),  # Identity transform
            interpolation,
            image.GetOrigin(),
            (min_spacing,) * len(original_spacing),
            image.GetDirection(),
            0,  # Default pixel value
            image.GetPixelID(),
        )
        if clip_negative:
            resampled_image = resampled_image*sitk.Cast(resampled_image>0, resampled_image.GetPixelID())    
    # Duplicate the outer slice of the image twice to pad it.
    dim = image.GetDimension()
    resampled_image = sitk.MirrorPad(
        sitk.MirrorPad(resampled_image, [1] * dim, [1] * dim), [1] * dim, [1] * dim
    )
    return resampled_image


def voxelwise_mean(image_list):
    # Convert SimpleITK to ITK
    image_array = [sitk.GetArrayViewFromImage(img) for img in image_list]

    # Use numpy for mean (parallel operations)
    # mean_array = np.mean(image_array, axis=0)
    mean_array = np.quantile(image_array, (0.2, 0.5, 0.8), axis=0).mean(axis=0)

    # Convert back
    mean_image = sitk.GetImageFromArray(mean_array)
    mean_image.CopyInformation(image_list[0])
    return mean_image


def voxelwise_std(image_list):
    # Convert SimpleITK to ITK
    image_array = [sitk.GetArrayViewFromImage(img) for img in image_list]

    # Use numpy for std (parallel operations)
    std_array = np.std(image_array, axis=0)

    # Convert back
    std_image = sitk.GetImageFromArray(std_array)
    std_image.CopyInformation(image_list[0])
    return std_image


def register_pair(
    fixed,
    moving,
    initial_transform=None,
    fixed_mask=None,
    level=3,
):
    """Register moving image to fixed image using Euler3DTransform."""
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric and sampling
    # registration_method.SetMetricAsCorrelation()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=32)
    # registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    # registration_method.SetMetricSamplingPercentage(0.95)
    registration_method.MetricUseFixedImageGradientFilterOn()
    registration_method.MetricUseMovingImageGradientFilterOn()

    if level==4:
        registration_method.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=0.1,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            estimateLearningRate=registration_method.Once,
            lineSearchUpperLimit=2.0,
            lineSearchEpsilon=0.1,
            maximumStepSizeInPhysicalUnits=fixed.GetSpacing()[0],
        )
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 2])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0.424628, 0])
    elif level==3:
        registration_method.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=0.1,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            estimateLearningRate=registration_method.Once,
            lineSearchUpperLimit=2.0,
            lineSearchEpsilon=0.1,
            maximumStepSizeInPhysicalUnits=fixed.GetSpacing()[0]*4,
        )
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 4, 4])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[
                0.424628 * 8,
                0.424628 * 4,
                0.424628 * 2,
                0.424628,
            ]
        )
    else:
        if level==2:
            num_iter = 20
        elif level==1:
            num_iter = 5
        elif level==0:
            num_iter = 1
        else:
            raise ValueError(f"The input {level} is invalid for level parameter.")
        registration_method.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=1.0,
            numberOfIterations=num_iter,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            estimateLearningRate=registration_method.EachIteration,
            lineSearchUpperLimit=5.0,
            maximumStepSizeInPhysicalUnits=fixed.GetSpacing()[0],
        )
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 4, 4])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[
                0.424628 * 8,
                0.424628 * 4,
                0.424628 * 2,
                0.424628,
            ]
        )

    registration_method.SetOptimizerScalesFromIndexShift()

    if not initial_transform:
        # A good estimate of the center-of-rotation is essential here
        # we don't want to be biased by activation or ventricular signal
        # so we use our otsu binary mask to find the COM
        binary_mask = make_mask(fixed)
        com_initializer = sitk.CenteredTransformInitializer(
            sitk.Cast(binary_mask, sitk.sitkFloat32),
            sitk.Cast(moving, sitk.sitkFloat32),
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        initial_transform = sitk.Euler3DTransform()
        initial_transform.SetCenter(com_initializer.GetCenter())

    # Initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    if fixed_mask:
        registration_method.SetMetricFixedMask(fixed_mask)

    # registration_method.AddCommand(
    #     sitk.sitkIterationEvent, lambda: command_iteration(registration_method)
    # )

    # Execute registration, pull out Euler3DTransform from wrapper
    return registration_method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    ).GetBackTransform()


def register_slice_pair(fixed, moving, slice_direction=2, interpolation=sitk.sitkBSpline5):
    """
    Registers each slice of the moving image to the corresponding slice of the fixed image.
    Assumes moving volume has already been registered to the fixed using register_pair
    and appropriately resampled.
    """
    fixed_size = fixed.GetSize()
    moving_size = moving.GetSize()

    num_slices = fixed_size[slice_direction]
    slice_transforms = []

    for z in range(num_slices):
        # Extract 2D slices
        if slice_direction == 2:
            fixed_slice = fixed[:, :, z]
            moving_slice = moving[:, :, z]
        elif slice_direction == 1:
            fixed_slice = fixed[:, z, :]
            moving_slice = moving[:, z, :]
        elif slice_direction == 0:
            fixed_slice = fixed[z, :, :]
            moving_slice = moving[z, :, :]

        fixed_slice = isotropic_upsample_and_pad(
            fixed_slice, interpolation=interpolation
        )

        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric and sampling
        # registration_method.SetMetricAsCorrelation()
        # registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=20)
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20)
        registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
        # registration_method.SetMetricSamplingPercentage(0.95)
        registration_method.MetricUseFixedImageGradientFilterOn()
        registration_method.MetricUseMovingImageGradientFilterOn()

        # Optimizer
        registration_method.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=0.1,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
            estimateLearningRate=registration_method.Once,
            lineSearchUpperLimit=2.0,
            lineSearchEpsilon=0.1,
            maximumStepSizeInPhysicalUnits=fixed_slice.GetSpacing()[0],
        )
        registration_method.SetOptimizerScalesFromIndexShift()

        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 2, 1, 1, 1])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[
                0.424628 * 8,
                0.424628 * 4,
                0.424628 * 2,
                0.424628,
                0,
            ]
        )

        # A good estimate of the center-of-rotation is essential here
        # we don't want to be biased by activation or ventricular signal
        # so we use our otsu binary mask to find the COM
        binary_mask = make_mask(fixed_slice)
        com_initializer = sitk.CenteredTransformInitializer(
            sitk.Cast(binary_mask, sitk.sitkFloat32),
            sitk.Cast(moving_slice, sitk.sitkFloat32),
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        initial_transform = sitk.Euler2DTransform()
        initial_transform.SetCenter(com_initializer.GetCenter())

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration
        # If the registration fails, return the identity transform
        try:
            final_transform = registration_method.Execute(
                sitk.Cast(fixed_slice, sitk.sitkFloat32),
                sitk.Cast(moving_slice, sitk.sitkFloat32),
            ).GetBackTransform()
        except:
            print("Slicewise registration exception, returning identity transform")
            final_transform = sitk.Euler2DTransform()
        slice_transforms.append(final_transform)

    return slice_transforms


def resample_slice_pair(
    reference, moving, transforms, slice_direction=2, interp=sitk.sitkBSpline5, clip_negative=True, extrapolator=True
):
    """
    Resamples the moving image to the reference image slice-by-slice using the provided transforms.

    Parameters:
        reference (sitk.Image): Reference 3D image.
        moving (sitk.Image): Moving 3D image.
        transforms (list): List of sitk.Transform objects (one per slice).
        interp: SimpleITK interpolation method.

    Returns:
        sitk.Image: Resampled 3D image.
    """
    fixed_size = reference.GetSize()
    moving_size = moving.GetSize()

    if fixed_size[slice_direction] != moving_size[slice_direction]:
        raise ValueError(
            "Reference and moving images must have the same number of slices."
        )
    if len(transforms) != fixed_size[slice_direction]:
        raise ValueError("Number of transforms must match number of slices.")

    num_slices = fixed_size[slice_direction]
    resampled_slices = []

    output_image = reference

    for z in range(num_slices):
        # Extract 2D slices
        if slice_direction == 2:
            fixed_slice = reference[:, :, z]
            moving_slice = moving[:, :, z]
        elif slice_direction == 1:
            fixed_slice = reference[:, z, :]
            moving_slice = moving[:, z, :]
        elif slice_direction == 0:
            fixed_slice = reference[z, :, :]
            moving_slice = moving[z, :, :]
        transform = transforms[z]

        resampled_slice = sitk.Resample(
            moving_slice,
            fixed_slice,
            transform,
            interp,
            0.0,
            reference.GetPixelID(),
            useNearestNeighborExtrapolator=extrapolator
        )
        if clip_negative:
            # set all negative values to 0
            resampled_slice = resampled_slice*sitk.Cast(resampled_slice>0, resampled_slice.GetPixelID())    
        if slice_direction == 2:
            output_image[:, :, z] = resampled_slice
        elif slice_direction == 1:
            output_image[:, z, :] = resampled_slice
        elif slice_direction == 0:
            output_image[z, :, :] = resampled_slice

    return output_image


def framewise_register_pair(moving_img, ref_img, level=1,interpolation=sitk.sitkBSpline5, max_workers=os.cpu_count()):
    # the input can be either a nifti file or an SITK image
    if isinstance(moving_img, sitk.Image):
        moving_img = moving_img
    elif os.path.isfile(moving_img):
        moving_img = sitk.ReadImage(moving_img, sitk.sitkFloat32)
    else:
        raise ValueError(f'{moving_img} is neither a file nor an SITK image.')

    # the input can be either a nifti file or an SITK image
    if isinstance(ref_img, sitk.Image):
        ref_img = ref_img
    elif os.path.isfile(ref_img):
        ref_img = sitk.ReadImage(ref_img, sitk.sitkFloat32)
    else:
        raise ValueError(f'{ref_img} is neither a file nor an SITK image.')

    if not ref_img.GetDimension()==3:
        raise ValueError(f"ref_img input must be 3D, got {ref_img.GetDimension()}D instead")

    if not moving_img.GetDimension()==4:
        raise ValueError(f"moving_img input must be 4D, got {moving_img.GetDimension()}D instead")

    fixed_upsample = isotropic_upsample_and_pad(ref_img, interpolation)

    size_4d = moving_img.GetSize()
    num_volumes = size_4d[3]

    # Extract 3D volumes
    volumes = []
    for i in range(num_volumes):
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size_4d[0], size_4d[1], size_4d[2], 0])
        extractor.SetIndex([0, 0, 0, i])
        volumes.append(extractor.Execute(moving_img))

    del extractor

    transforms = [None] * num_volumes
    # Parallel Registration
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=num_volumes) as pbar:
            futures = {}
            for i in range(0, num_volumes):
                future = executor.submit(
                    register_pair,
                    fixed=fixed_upsample,
                    moving=volumes[i],
                    fixed_mask=None,
                    level=level,
                )
                futures[future] = i
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                transforms[i] = future.result()
                pbar.update(1)
    return transforms


def main(input_file, output_prefix, slice_moco=False, two_pass_slice_moco=False):
    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load 4D fMRI data
    fmri_4d = sitk.ReadImage(input_file)
    size_4d = fmri_4d.GetSize()
    num_volumes = size_4d[3]

    # Extract 3D volumes
    volumes = []
    for i in range(num_volumes):
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size_4d[0], size_4d[1], size_4d[2], 0])
        extractor.SetIndex([0, 0, 0, i])
        volumes.append(extractor.Execute(fmri_4d))

    del fmri_4d  # Free memory
    del extractor

    # Determine middle volume index
    mid_idx = num_volumes // 2

    fixed_upsample = isotropic_upsample_and_pad(volumes[mid_idx], sitk.sitkBSpline5)
    # fixed_mask = make_mask(fixed_upsample)
    # statsfilter = sitk.LabelShapeStatisticsImageFilter()
    # statsfilter.Execute(fixed_mask)
    # fixed_centre = statsfilter.GetCentroid(1)

    transforms = [None] * num_volumes
    # Even though the initalization will be a identity transform, we need the center of rotation
    transforms[mid_idx] = sitk.CenteredTransformInitializer(
        volumes[mid_idx],
        volumes[mid_idx],
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    round = 0

    print(f"Registering time slices to volume {mid_idx + 1}")
    # Parallel Registration
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=num_volumes - 1) as pbar:
            futures = {}
            for i in range(0, num_volumes):
                if i == mid_idx:
                    continue
                future = executor.submit(
                    register_pair,
                    fixed=fixed_upsample,
                    moving=volumes[i],
                    fixed_mask=None,
                )
                futures[future] = i
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                transforms[i] = future.result()
                pbar.update(1)

    write_transforms_to_csv(transforms, output_prefix + f"moco{round}.csv")

    print("Resampling time slices to middle volume")
    # Parallel Resampling
    resampled = [None] * num_volumes
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=num_volumes - 1) as pbar:
            futures = {}
            for i in range(0, num_volumes):
                if i == mid_idx:
                    resampled[i] = sitk.Cast(volumes[i], sitk.sitkFloat32)
                    continue
                future = executor.submit(
                    resample_volume, volumes[i], volumes[mid_idx], transforms[i]
                )
                futures[future] = i
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                resampled[i] = future.result()
                pbar.update(1)

    print("Calculating mean of corrected 3D volumes")
    mean_image = voxelwise_mean(resampled)
    moco_timeseries = sitk.JoinSeries(resampled)
    moco_std = voxelwise_std(resampled)
    sitk.WriteImage(mean_image, output_prefix + f"mean{round}.nii.gz")
    sitk.WriteImage(moco_timeseries, output_prefix + f"moco{round}.nii.gz")
    sitk.WriteImage(moco_std, output_prefix + f"stddev{round}.nii.gz")
    round += 1

    ################################################################
    # Slice-by-slice registration
    ################################################################

    slicewise_transforms = [None] * num_volumes
    slicewise_resampled = [None] * num_volumes

    if slice_moco or two_pass_slice_moco:
        print(
            "Back projecting mean image to individual volumes and performing slice-by-slice registration"
        )
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with tqdm(total=num_volumes) as pbar:
                futures = {}
                for i in range(0, num_volumes):
                    future = executor.submit(
                        register_slice_pair,
                        fixed=resample_volume(
                            mean_image, volumes[i], transforms[i].GetInverse()
                        ),
                        moving=volumes[i],
                    )
                    futures[future] = i
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    slicewise_transforms[i] = future.result()
                    pbar.update(1)

        print("Resampling 2D slices")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with tqdm(total=num_volumes) as pbar:
                futures = {}
                for i in range(0, num_volumes):
                    future = executor.submit(
                        resample_slice_pair,
                        reference=volumes[i],
                        moving=volumes[i],
                        transforms=slicewise_transforms[i],
                    )
                    futures[future] = i
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    slicewise_resampled[i] = future.result()
                    pbar.update(1)

        # Parallel Resampling
        print("Resampling corrected 3D volumes to middle volume")
        resampled = [None] * num_volumes
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with tqdm(total=num_volumes - 1) as pbar:
                futures = {}
                for i in range(0, num_volumes):
                    if i == mid_idx:
                        resampled[i] = sitk.Cast(
                            slicewise_resampled[i], sitk.sitkFloat32
                        )
                        continue
                    future = executor.submit(
                        resample_volume,
                        slicewise_resampled[i],
                        volumes[mid_idx],
                        transforms[i],
                    )
                    futures[future] = i
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    resampled[i] = future.result()
                    pbar.update(1)

        print("Calculating mean of corrected 3D volumes")
        mean_image = voxelwise_mean(resampled)
        moco_timeseries = sitk.JoinSeries(resampled)
        moco_std = voxelwise_std(resampled)
        sitk.WriteImage(mean_image, output_prefix + f"mean{round}.nii.gz")
        sitk.WriteImage(moco_timeseries, output_prefix + f"moco{round}.nii.gz")
        sitk.WriteImage(moco_std, output_prefix + f"stddev{round}.nii.gz")
        round += 1

        if two_pass_slice_moco:
            print("Performing second-pass slice-by-slice registration")
            fixed_upsample = isotropic_upsample_and_pad(mean_image, sitk.sitkBSpline5)
            print("Registering volumes to mean image")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                with tqdm(total=num_volumes) as pbar:
                    futures = {}
                    for i in range(0, num_volumes):
                        future = executor.submit(
                            register_pair,
                            fixed=fixed_upsample,
                            moving=slicewise_resampled[i],
                            initial_transform=transforms[i],
                        )
                        futures[future] = i
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        transforms[i] = future.result()
                        pbar.update(1)

            write_transforms_to_csv(transforms, output_prefix + f"moco{round}.csv")

            print("Resampling volumes to mean image")
            resampled = [None] * num_volumes
            with concurrent.futures.ProcessPoolExecutor() as executor:
                with tqdm(total=num_volumes) as pbar:
                    futures = {}
                    for i in range(0, num_volumes):
                        future = executor.submit(
                            resample_volume,
                            volume=slicewise_resampled[i],
                            reference=mean_image,
                            transform=transforms[i],
                            interpolation=sitk.sitkBSpline5,
                        )
                        futures[future] = i
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        resampled[i] = future.result()
                        pbar.update(1)

            print("Calculating mean of corrected 3D volumes")
            mean_image = voxelwise_mean(resampled)
            moco_timeseries = sitk.JoinSeries(resampled)
            moco_std = voxelwise_std(resampled)
            sitk.WriteImage(mean_image, output_prefix + f"mean{round}.nii.gz")
            sitk.WriteImage(moco_timeseries, output_prefix + f"moco{round}.nii.gz")
            sitk.WriteImage(moco_std, output_prefix + f"stddev{round}.nii.gz")
            round += 1

            print(
                "Back projecting mean image to individual volumes and performing slice-by-slice registration"
            )
            with concurrent.futures.ProcessPoolExecutor() as executor:
                with tqdm(total=num_volumes) as pbar:
                    futures = {}
                    for i in range(0, num_volumes):
                        future = executor.submit(
                            register_slice_pair,
                            fixed=resample_volume(
                                mean_image, volumes[i], transforms[i].GetInverse()
                            ),
                            moving=volumes[i],
                        )
                        futures[future] = i
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        slicewise_transforms[i] = future.result()
                        pbar.update(1)

            print("Resampling 2D slices")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                with tqdm(total=num_volumes) as pbar:
                    futures = {}
                    for i in range(0, num_volumes):
                        future = executor.submit(
                            resample_slice_pair,
                            reference=volumes[i],
                            moving=volumes[i],
                            transforms=slicewise_transforms[i],
                        )
                        futures[future] = i
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        slicewise_resampled[i] = future.result()
                        pbar.update(1)

            # Parallel Resampling
            print("Resampling corrected 3D volumes to mean volume")
            resampled = [None] * num_volumes
            with concurrent.futures.ProcessPoolExecutor() as executor:
                with tqdm(total=num_volumes) as pbar:
                    futures = {}
                    for i in range(0, num_volumes):
                        future = executor.submit(
                            resample_volume,
                            slicewise_resampled[i],
                            mean_image,
                            transforms[i],
                        )
                        futures[future] = i
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        resampled[i] = future.result()
                        pbar.update(1)

            print("Calculating mean of corrected 3D volumes")
            mean_image = voxelwise_mean(resampled)
            moco_timeseries = sitk.JoinSeries(resampled)
            moco_std = voxelwise_std(resampled)
            sitk.WriteImage(mean_image, output_prefix + f"mean{round}.nii.gz")
            sitk.WriteImage(moco_timeseries, output_prefix + f"moco{round}.nii.gz")
            sitk.WriteImage(moco_std, output_prefix + f"stddev{round}.nii.gz")
            round += 1
    else:
        slicewise_resampled = volumes

    ################################################################
    # Registration Round 3
    ################################################################
    # Upsample new mean to isotropic and pad
    fixed_upsample = isotropic_upsample_and_pad(mean_image, sitk.sitkBSpline5)
    # fixed_mask = make_mask(fixed_upsample)

    print("Registering volumes to mean image")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=num_volumes) as pbar:
            futures = {}
            for i in range(0, num_volumes):
                future = executor.submit(
                    register_pair,
                    fixed=fixed_upsample,
                    moving=slicewise_resampled[i],
                    initial_transform=transforms[i],
                    fixed_mask=None,
                    level=4,
                )
                futures[future] = i
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                transforms[i] = future.result()
                pbar.update(1)

    write_transforms_to_csv(transforms, output_prefix + f"moco{round}.csv")

    print("Resampling volumes to mean image")
    resampled = [None] * num_volumes
    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=num_volumes) as pbar:
            futures = {}
            for i in range(0, num_volumes):
                future = executor.submit(
                    resample_volume,
                    volume=slicewise_resampled[i],
                    reference=mean_image,
                    transform=transforms[i],
                    interpolation=sitk.sitkBSpline5,
                )
                futures[future] = i
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                resampled[i] = future.result()
                pbar.update(1)

    print("Calculating mean of corrected 3D volumes")
    mean_image = voxelwise_mean(resampled)
    moco_timeseries = sitk.JoinSeries(resampled)
    moco_std = voxelwise_std(resampled)
    sitk.WriteImage(mean_image, output_prefix + f"mean{round}.nii.gz")
    sitk.WriteImage(moco_timeseries, output_prefix + f"moco{round}.nii.gz")
    sitk.WriteImage(moco_std, output_prefix + f"stddev{round}.nii.gz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="fMRI motion correction")
    parser.add_argument("input", help="Input 4D fMRI file (NIfTI format)")
    parser.add_argument("output_prefix", help="Output prefix")
    parser.add_argument(
        "--slice-moco",
        action="store_true",
        help="Enable 2D slice-by-slice motion correction",
    )
    parser.add_argument(
        "--two-pass-slice-moco",
        action="store_true",
        help="Enable two-pass 2D slice-by-slice motion correction",
    )
    args = parser.parse_args()

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    print(f"Processing {args.input}")
    main(args.input, args.output_prefix, args.slice_moco, args.two_pass_slice_moco)
