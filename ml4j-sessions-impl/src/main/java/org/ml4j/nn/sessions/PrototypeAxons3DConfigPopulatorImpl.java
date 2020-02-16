/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.sessions;

import org.ml4j.nn.axons.Axons3DConfigBuilderBase;
import org.ml4j.nn.axons.Axons3DConfigPopulator;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Prototype implementation of Axons3DConfigPopulator - TODO -refactor.
 * 
 * @author Michael Lavelle
 *
 */
public class PrototypeAxons3DConfigPopulatorImpl<B extends Axons3DConfigBuilderBase<?, ?>> implements Axons3DConfigPopulator<B> {

	public Neurons3D getCalculatedRightNeurons(Neurons3D leftNeurons, int filterWidth, int filterHeight,
			int paddingWidth, int paddingHeight, int strideWidth, int strideHeight, int outputDepth,
			boolean outputBiasUnit) {

		int filterWidthMinusInputWidthWithPadding = filterWidth - leftNeurons.getWidth() + paddingWidth * 2;
		int filterWidthMinusInputWidthWithPaddingDividedByStrideWidth = filterWidthMinusInputWidthWithPadding
				/ strideWidth;

		if (filterWidthMinusInputWidthWithPaddingDividedByStrideWidth
				* strideWidth != filterWidthMinusInputWidthWithPadding) {
			throw new IllegalStateException(
					"The input width with padding is not an integer multiple of the stride width");
		}

		int rightNeuronsWidth = 1 - filterWidthMinusInputWidthWithPaddingDividedByStrideWidth;
		int filterHeightMinusInputHeightWithPadding = filterHeight - leftNeurons.getHeight() + paddingHeight * 2;
		int filterHeightMinusInputHeightWithPaddingDividedByStrideHeight = filterHeightMinusInputHeightWithPadding
				/ strideHeight;

		if (filterHeightMinusInputHeightWithPaddingDividedByStrideHeight
				* strideHeight != filterHeightMinusInputHeightWithPadding) {
			throw new IllegalStateException(
					"The input height with padding is not an integer multiple of the stride height");
		}

		int rightNeuronsHeight = 1 - filterHeightMinusInputHeightWithPaddingDividedByStrideHeight;

		if (rightNeuronsWidth < 1 || rightNeuronsHeight < 1) {
			throw new IllegalStateException("Invalid configuration - neither right neurons have been specified "
					+ "and the calculated right neurons base on the specified parameters would have dimensions < 1");
		} else {
			return new Neurons3D(rightNeuronsWidth, rightNeuronsHeight, outputDepth, outputBiasUnit);
		}

	}

	public int getCalculatedWidthForSamePadding(Neurons3D leftNeurons, int finalFilterWidth, int strideWidth,
			Integer explicitlySetPaddingWidth) {

		// P = ((S-1)*W-S+F)/2
		float calculatedPaddingWidth = ((float) ((strideWidth - 1) * leftNeurons.getWidth() - strideWidth
				+ finalFilterWidth)) / 2f;
		int calculatedPaddingWidthInt = (int) calculatedPaddingWidth;

		if (calculatedPaddingWidth != calculatedPaddingWidthInt) {
			calculatedPaddingWidthInt = calculatedPaddingWidthInt + 1;
		}

		if (explicitlySetPaddingWidth != null && explicitlySetPaddingWidth != calculatedPaddingWidthInt) {
			throw new IllegalStateException(
					"Invalid configuration - explicitly set padding width does not match the padding width "
							+ "required for samePadding");
		}

		return calculatedPaddingWidthInt;
	}

	public int getCalculatedHeightForSamePadding(Neurons3D leftNeurons, int finalFilterHeight, int strideHeight,
			Integer explicitlySetPaddingHeight) {

		// P = ((S-1)*W-S+F)/2
		float calculatedPaddingHeight = ((float) ((strideHeight - 1) * leftNeurons.getHeight() - strideHeight
				+ finalFilterHeight)) / 2f;
		int calculatedPaddingHeightInt = (int) calculatedPaddingHeight;

		if (calculatedPaddingHeight != calculatedPaddingHeightInt) {
			calculatedPaddingHeightInt = calculatedPaddingHeightInt + 1;
		}

		if (explicitlySetPaddingHeight != null && explicitlySetPaddingHeight != calculatedPaddingHeightInt) {
			throw new IllegalStateException(
					"Invalid configuration - explicitly set padding height does not match the padding height "
							+ "required for samePadding");
		}

		return calculatedPaddingHeightInt;
	}

	// TODO - refactor this method
	@Override
	public B populateAndValidate(B configBuilder) {

		Neurons3D leftNeurons = configBuilder.getLeftNeurons();

		if (leftNeurons == null) {
			throw new IllegalStateException("Input neurons cannot be null");
		}

		Neurons3D rightNeurons = configBuilder.getRightNeurons();
		Integer outputDepth = configBuilder.getOutputDepth();
		boolean outputBiasUnit = configBuilder.isOutputBiasUnit();
		Integer paddingWidth = configBuilder.getPaddingWidth();
		Integer paddingHeight = configBuilder.getPaddingHeight();
		int strideHeight = configBuilder.getStrideHeight();
		int strideWidth = configBuilder.getStrideWidth();
		Integer filterWidth = configBuilder.getFilterWidth();
		Integer filterHeight = configBuilder.getFilterHeight();
		Boolean samePadding = configBuilder.getSamePadding();

		Integer localFilterWidth = null;
		Integer localFilterHeight = null;
		Integer localPaddingWidth = null;
		Integer localPaddingHeight = null;
		Neurons3D localRightNeurons = null;

		// Check consistency of rightNeurons and output depth and output bias unit ( if set)
		if (rightNeurons != null) {
			if (outputDepth != null && outputDepth != rightNeurons.getDepth()) {
				throw new IllegalStateException(
						"Inconsistency between explicitly set output depth and right neurons depth");
			}
			if (outputBiasUnit != rightNeurons.hasBiasUnit()) {
				throw new IllegalStateException(
						"Inconsistency between explicitly set output bias unit and right neurons bias unit");
			}
		} else {
			if (outputDepth == null) {
				throw new IllegalStateException("Unable to determine filter count");
			}
		}

		// Assume strideWidth and strideHeight have been set, or have been defaulted to
		// 1.

		// Left neurons, strideWidth and strideHeight are all set.
		// Need to set right neurons, paddingheight, padding width, filter width, filter
		// height.

		if (rightNeurons != null) {

			localRightNeurons = rightNeurons;

			// Left neurons, right neurons, strideWidth and strideHeight are all set.
			// Need to set paddingheight, padding width, filter width, filter height.

			if (paddingWidth != null && paddingHeight != null) {

				// Left neurons, right neurons, strideWidth and strideHeight, padding width,
				// padding height are all set.
				// Need to set filter width, filter height.

				// Get calculated filter heights/widths - also validates these values match any
				// explicitly set height/widths
				int finalFilterWidth = getCalculatedFilterWidth(leftNeurons, rightNeurons, paddingWidth, strideWidth,
						filterWidth);
				int finalFilterHeight = getCalculatedFilterHeight(leftNeurons, rightNeurons, paddingHeight,
						strideHeight, filterHeight);

				if (samePadding != null && samePadding.booleanValue()) {

					// Validates that the calculated same padding matches the explicitly set values.

					localPaddingWidth = getCalculatedWidthForSamePadding(leftNeurons, finalFilterWidth, strideWidth,
							paddingWidth);
					localPaddingHeight = getCalculatedWidthForSamePadding(leftNeurons, finalFilterHeight, strideHeight,
							paddingHeight);

				} else {
					localPaddingWidth = paddingWidth;
					localPaddingHeight = paddingHeight;
				}

				localFilterWidth = finalFilterWidth;
				localFilterHeight = finalFilterHeight;

			} else {
				// Padding width or height has not been set.

				// Need to set paddingheight, padding width, filter width, filter height
				if (samePadding != null && samePadding.booleanValue()) {

					if (filterHeight != null && filterWidth != null) {
						// Need to set paddingheight, padding width.

						localPaddingWidth = getCalculatedWidthForSamePadding(leftNeurons, filterWidth, strideWidth,
								paddingWidth);
						localPaddingHeight = getCalculatedWidthForSamePadding(leftNeurons, filterHeight, strideHeight,
								paddingHeight);

						localFilterWidth = filterWidth;
						localFilterHeight = filterHeight;

					} else {
						throw new IllegalStateException(
								"Unable to configure for same padding as right neurons have not been set and "
										+ "one of filterWidth/filterHeight has not been set");
					}

				} else {

					// Left neurons, right neurons, strideWidth and strideHeight are all set.

					// Need to set paddingheight, padding width, filter width, filter height

					if (paddingHeight == null && filterHeight == null) {
						// Default paddingHeight to zero
						// Calculate filterHeight.
						localPaddingHeight = 0;
						localFilterHeight = getCalculatedFilterHeight(leftNeurons, rightNeurons, localPaddingHeight,
								strideHeight, filterHeight);
					} else if (paddingHeight != null) {

						// Validates that the filter height matches the calculated filter height
						localFilterHeight = getCalculatedFilterHeight(leftNeurons, rightNeurons, paddingHeight,
								strideHeight, filterHeight);

						localPaddingHeight = paddingHeight;

					} else {

						int inputHeightWithPadding = filterHeight - (1 - rightNeurons.getHeight()) * strideHeight;

						int paddingHeightTimesTwo = inputHeightWithPadding - leftNeurons.getHeight();
						int calculatedPaddingHeight = paddingHeightTimesTwo / 2;
						if (calculatedPaddingHeight * 2 != paddingHeightTimesTwo) {
							throw new IllegalStateException("Padding height calculation rounding error");
						} else {
							localPaddingHeight = calculatedPaddingHeight;
						}

						localFilterHeight = filterHeight;

					}

					if (paddingWidth == null && filterWidth == null) {
						// Default paddingWidth to zero
						// Calculate filterWidth.

						localPaddingWidth = 0;
						localFilterWidth = getCalculatedFilterWidth(leftNeurons, rightNeurons, localPaddingWidth,
								strideWidth, filterWidth);

					} else if (paddingWidth != null) {
						// Validates that the filter height matches the calculated filter height
						localFilterWidth = getCalculatedFilterWidth(leftNeurons, rightNeurons, paddingWidth,
								strideWidth, filterWidth);

						localPaddingWidth = paddingWidth;
					}	else {

						int inputWidthWithPadding = filterWidth - (1 - rightNeurons.getWidth()) * strideWidth;

						int paddingWidthTimesTwo = inputWidthWithPadding - leftNeurons.getWidth();
						int calculatedPaddingWidth = paddingWidthTimesTwo / 2;
						if (calculatedPaddingWidth * 2 != paddingWidthTimesTwo) {
							throw new IllegalStateException("Padding width calculation rounding error");
						} else {
							localPaddingWidth = calculatedPaddingWidth;
						}

						localFilterWidth = filterWidth;

					}

				}

			}

		} else {
			// Right neurons has not been set

			// Left neurons, strideWidth and strideHeight are all set.
			// Need to set right neurons, paddingheight, padding width, filter width, filter
			// height.

			// Need filter width and filter height to both be specified, or we won't be able
			// to calculate them

			if (filterWidth == null || filterHeight == null) {
				throw new IllegalStateException(
						"Unable to calculate filter size as right neurons have not been set and "
								+ "one of filterWidth/filterHeight has not been set");
			} else {

				localFilterWidth = filterWidth;
				localFilterHeight = filterHeight;

				// Left neurons, strideWidth and strideHeight, filterWidth, filterHeight are all
				// set.
				// Need to set right neurons, paddingheight, padding width
				if (samePadding) {

					localPaddingWidth = getCalculatedWidthForSamePadding(leftNeurons, filterWidth, strideWidth,
							paddingWidth);
					localPaddingHeight = getCalculatedWidthForSamePadding(leftNeurons, filterHeight, strideHeight,
							paddingHeight);

					// Left neurons, strideWidth and strideHeight, filterWidth, filterHeight,
					// padding width, padding height are all set.
					// Need to set right neurons

					if (outputDepth == null) {
						throw new IllegalStateException("Unable to determine output filter count");
					} else {
						localRightNeurons = getCalculatedRightNeurons(leftNeurons, filterWidth, filterHeight,
								paddingWidth, paddingHeight, strideWidth, strideHeight, outputDepth, outputBiasUnit);
					}

				} else {

					if (paddingWidth == null) {
						// Default to zero
						localPaddingWidth = 0;
					} else {
						localPaddingWidth = paddingWidth;
					}
					if (paddingHeight == null) {
						// Default to zero
						localPaddingHeight = 0;
					} else {
						localPaddingHeight = paddingHeight;
					}
					// Left neurons, strideWidth and strideHeight, filterWidth, filterHeight,
					// padding width, padding height are all set.

					// Need to set right neurons,

					if (outputDepth == null) {
						throw new IllegalStateException("Unable to determine output filter count");
					} else {
						localRightNeurons = getCalculatedRightNeurons(leftNeurons, filterWidth, filterHeight,
								localPaddingWidth, localPaddingHeight, strideWidth, strideHeight, outputDepth,
								outputBiasUnit);
					}

				}

			}
		}

		if (samePadding != null && samePadding.booleanValue() && (leftNeurons.getHeight() != localRightNeurons.getHeight()
				|| leftNeurons.getWidth() != localRightNeurons.getWidth())) {
			throw new IllegalStateException(
					"For same padding, the left neurons and right neurons must have the same width/height");
		}

		configBuilder.withOutputNeurons(localRightNeurons);
		configBuilder.withFilterWidth(localFilterWidth);
		configBuilder.withFilterHeight(localFilterHeight);

		configBuilder.withPaddingHeight(localPaddingHeight);
		configBuilder.withPaddingWidth(localPaddingWidth);

		return configBuilder;

	}

	public int getCalculatedFilterWidth(Neurons3D leftNeurons, Neurons3D rightNeurons, int paddingWidth,
			int strideWidth, Integer explicitlySetFilterWidth) {
		
		int inputWidthWithPadding = leftNeurons.getWidth() + paddingWidth * 2;

		int calculatedFilterWidth = inputWidthWithPadding + (1 - rightNeurons.getWidth()) * (strideWidth);

		if (calculatedFilterWidth < 1) {
			throw new IllegalStateException("Invalid configuration - calculated filter width cannot be less than 1");
		}

		if (explicitlySetFilterWidth != null && explicitlySetFilterWidth.intValue() != calculatedFilterWidth) {
			throw new IllegalStateException("Explicitly set filter width of:" + explicitlySetFilterWidth
					+ " is inconsistent with calculated filter width of:" + calculatedFilterWidth);
		}

		return calculatedFilterWidth;
	}

	public int getCalculatedFilterHeight(Neurons3D leftNeurons, Neurons3D rightNeurons, int paddingHeight,
			int strideHeight, Integer explicitlySetFilterHeight) {

		int inputHeightWithPadding = leftNeurons.getHeight() + paddingHeight * 2;

		int calculatedFilterHeight = inputHeightWithPadding + (1 - rightNeurons.getHeight()) * (strideHeight);

		if (calculatedFilterHeight < 1) {
			throw new IllegalStateException("Invalid configuration - calculated filter height cannot be less than 1");
		}

		if (explicitlySetFilterHeight != null && explicitlySetFilterHeight.intValue() != calculatedFilterHeight) {
			throw new IllegalStateException("Explicitly set filter height of:" + explicitlySetFilterHeight
					+ " is inconsistent with calculated filter height of:" + calculatedFilterHeight);
		}

		return calculatedFilterHeight;
	}

}
