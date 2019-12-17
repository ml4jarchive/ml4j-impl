package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class DummyAxonsFactoryImpl implements AxonsFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	
	public DummyAxonsFactoryImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}
	
	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons, Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth,  int strideHeight, int paddingWidth,
			int paddingHeight) {
		return new DummyAveragePoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, strideWidth, strideHeight);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth,
			Integer paddingHeight, Matrix connectionWeights, Matrix biases) {
		return new DummyConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, strideWidth, strideHeight, paddingHeight, paddingHeight);
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, boolean scaleOutputs, int strideWidth,  int strideHeight, int paddingWidth,
			int paddingHeight) {
		return new DummyMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, strideWidth, strideHeight);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(N leftNeurons, N rightNeurons, Matrix gamma,
			Matrix beta) {
		return new DummyScaleAndShiftAxonsImpl<>(matrixFactory, leftNeurons, rightNeurons);
	}

}
