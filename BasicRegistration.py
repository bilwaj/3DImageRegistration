# Please note that
# SimpleITK 0.9 + is needed for this program to work
import SimpleITK as sitk 
import numpy as np 
import os
#read the images
 
def register_linearly(fixed_image,moving_image):
#initial alignment of the two volumes
	transform = sitk.CenteredTransformInitializer(fixed_image, 
	                                              moving_image, 
	                                              sitk.Euler3DTransform(), 
	                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
	 
	#multi-resolution rigid registration using Mutual Information
	toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
	toDisplacementFilter.SetReferenceImage(fixed_image)

	registration_method = sitk.ImageRegistrationMethod()
	registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
	registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
	registration_method.SetMetricSamplingPercentage(0.01)
	registration_method.SetInterpolator(sitk.sitkLinear)
	registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
	                                                  numberOfIterations=100, 
	                                                  convergenceMinimumValue=1e-6, 
	                                                  convergenceWindowSize=10)
	registration_method.SetOptimizerScalesFromPhysicalShift()
	registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
	registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
	registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
	registration_method.SetInitialTransform(transform)
	registration_method.Execute(fixed_image, moving_image)

	#displacementField = toDisplacementFilter.Execute(transform)
	moving_transformed = sitk.Resample(moving_image, fixed_image, transform, 
	                                   sitk.sitkLinear, 0.0, 
	                                   moving_image.GetPixelIDValue())
	return (transform,moving_transformed)

#Non-linear adjustments done using Demons. These are likely to be minor and the algorithm is 
#calibrated to make sure they stay small
def command_iteration(filter) :
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

def nonlinear_adjustments(fixed_image,moving_image):
	matcher = sitk.HistogramMatchingImageFilter()
	matcher.SetNumberOfHistogramLevels(1024)
	matcher.SetNumberOfMatchPoints(7)
	matcher.ThresholdAtMeanIntensityOn()
	moving = matcher.Execute(moving_image,fixed_image)

	demons = sitk.DemonsRegistrationFilter()
	demons.SetNumberOfIterations(50)
	demons.SetStandardDeviations(1.0)

	demons.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(demons))
	displacementField = demons.Execute( fixed_image, moving_image )
	outTx = sitk.DisplacementFieldTransform(displacementField)
	resampler = sitk.ResampleImageFilter()
 	resampler.SetReferenceImage(fixed_image);
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(outTx)
	out = resampler.Execute(moving_image)
	return out


def register(timepoint1,timepoint2):
	fixed_image =  sitk.ReadImage(timepoint1, sitk.sitkFloat32)
	moving_image = sitk.ReadImage(timepoint2, sitk.sitkFloat32)
	(transform,moving_transformed)=register_linearly(fixed_image,moving_image)
	out=nonlinear_adjustments(fixed_image,moving_transformed)
	sitk.WriteImage(out,'out.nii')
