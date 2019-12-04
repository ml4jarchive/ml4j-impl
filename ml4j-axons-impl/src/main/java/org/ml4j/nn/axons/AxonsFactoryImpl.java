package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class AxonsFactoryImpl implements AxonsFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;

	public AxonsFactoryImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			Matrix connectionWeights, Matrix biases) {
		
		if (connectionWeights != null) {
					if (biases != null) {
							return new FullyConnectedAxonsAlternateImpl(leftNeurons, rightNeurons, connectionWeights, biases);
					} else {
						return new FullyConnectedAxonsAlternateImpl(leftNeurons, rightNeurons, connectionWeights, null);
					
				}
		} else {
			return new FullyConnectedAxonsAlternateImpl(leftNeurons, rightNeurons, matrixFactory);
		}
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth,
			int strideHeight, Integer paddingWidth, Integer paddingHeight, Matrix connectionWeights, Matrix biases) {
		if (paddingWidth != null || paddingHeight != null) {
			if (connectionWeights != null) {
				if (leftNeurons.getWidth() == rightNeurons.getWidth() && leftNeurons.getHeight() == rightNeurons.getHeight()) {
					int inputWidth = leftNeurons.getWidth();
					int inputHeight = leftNeurons.getHeight();
					int outputWidth = rightNeurons.getWidth();
					int outputHeight = rightNeurons.getHeight();

					
					int inputWidthWithPadding = inputWidth + paddingWidth * 2;
					int inputHeightWithPadding = inputHeight + paddingHeight * 2;
					int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (strideWidth);
					int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (strideHeight);
					if (filterWidth != 1 && filterWidth == filterHeight && strideWidth == 1 && strideHeight == 1) {
						return new ConvolutionalAxonsAlternateImpl(this, matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
								.withStrideHeight(strideHeight).withPaddingWidth(paddingWidth).withPaddingHeight(paddingHeight), connectionWeights, biases == null ? null : biases, false);
					}
				}
					return new ConvolutionalAxonsAlternateImpl(this, matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
							.withStrideHeight(strideHeight).withPaddingWidth(paddingWidth).withPaddingHeight(paddingHeight), connectionWeights, biases == null ? null : biases, false);
			} else {
					return new ConvolutionalAxonsAlternateImpl(this, matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
							.withStrideHeight(strideHeight).withPaddingWidth(paddingWidth).withPaddingHeight(paddingHeight), false);
			}
		} else {
			if (connectionWeights != null) {
					return new ConvolutionalAxonsAlternateImpl(this, matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
							.withStrideHeight(strideHeight).withPaddingWidth(0).withPaddingHeight(0), connectionWeights.dup(), biases == null ? null : biases.dup(), false);
			} else {
					return new ConvolutionalAxonsAlternateImpl(this, matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
							.withStrideHeight(strideHeight).withPaddingWidth(0).withPaddingHeight(0), false);
			}
		}
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, boolean scaleOutputs, int strideWidth, int strideHeight, int paddingWidth, int paddingHeight) {
			return new MaxPoolingAxonsAlternateImpl(matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
					.withStrideHeight(strideHeight).withPaddingWidth(paddingWidth).withPaddingHeight(paddingHeight), scaleOutputs);

	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, int paddingWidth, int paddingHeight) {
			return new AveragePoolingAxonsAlternateImpl(matrixFactory, leftNeurons, rightNeurons, new Axons3DConfig().withStrideWidth(strideWidth)
					.withStrideHeight(strideHeight).withPaddingWidth(paddingWidth).withPaddingHeight(paddingHeight));

	}

	@SuppressWarnings("unchecked")
	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(N leftNeurons, N rightNeurons, Matrix gamma, Matrix beta) {
		Matrix initialGamma = gamma != null ? gamma : matrixFactory.createOnes(rightNeurons.getNeuronCountExcludingBias(), 1);
		Matrix initialBeta = beta != null ? beta : matrixFactory.createZeros(rightNeurons.getNeuronCountExcludingBias(), 1);
		ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(initialGamma, initialBeta);
		return new ScaleAndShiftAxonsAlternateImpl<>(leftNeurons, rightNeurons, (N) new Neurons3D(1,1,1,true), matrixFactory, config);
	}

}
