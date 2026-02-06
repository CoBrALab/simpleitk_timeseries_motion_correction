import math

def make_pyramid(
    min_spacing,
    min_length,
    final_iterations=50,
    rough=False,
    close=False,
):
    """
    Python equivalent of the bash function from https://github.com/CoBrALab/antsRegistration_affine_SyN/blob/master/antsRegistration_affine_SyN.sh
    """

    min_voxel_number = 8
    if not min_voxel_number<min_length: # this is necessary for max_shrink>0
        raise ValueError(f"At least {min_voxel_number+1} voxels are expected per axis, the input minimum length provided was {min_length}")
    if rough:
        if not min_voxel_number*3<=min_length: # this is necessary for max_octave>=min_octave
            raise ValueError(f"At least {min_voxel_number*3} voxels are expected per axis with rough=True, the input minimum length provided was {min_length}")
    
    max_shrink = int(round(min_length / (min_voxel_number*2)))

    max_octave = int(round(
        math.log(max_shrink) / math.log(2) + 0.55
    ))

    if close and max_octave > 2:
        max_octave = 2

    min_octave = 1
    if rough:
        min_octave = 2

    shrinks_l_l = []
    smooths_l_l = []
    iterations_l_l = []

    for octave in range(max_octave, min_octave - 1, -1):
        shrinks_l = []
        smooths_l = []
        iterations_l = []
        for scale in range(5, 0, -1):

            shrink = 2 ** octave

            if shrink >= max_shrink:
                shrinks_l.append(max_shrink)
            else:
                shrinks_l.append(shrink)

            # Compute the minimum safe blur for this octave
            # From https://discourse.itk.org/t/resampling-to-isotropic-signal-processing-theory/1403/14
            numerator = (min_spacing * shrink)**2 - min_spacing**2
            denominator = (2 * math.sqrt(2 * math.log(2))) ** 2
            sigma = math.sqrt(numerator / denominator)

            # Scale up smoothing
            smooth = sigma + sigma * (scale - 1) * math.sqrt(0.5)
            smooths_l.append(smooth)

            # Iterations per level
            it = int(final_iterations * (octave + 1) * math.sqrt(2))
            iterations_l.append(it)

        shrinks_l_l.append(shrinks_l)
        smooths_l_l.append(smooths_l)
        iterations_l_l.append(iterations_l)
        
    # Finest level (if not rough)
    if not rough:
        shrinks_l_l.append([1])
        smooths_l_l.append([0.0])
        iterations_l_l.append([final_iterations])
    return shrinks_l_l, smooths_l_l, iterations_l_l