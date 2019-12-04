package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class DummyDirectedAxonsComponentFactory implements DirectedAxonsComponentFactory {

	private MatrixFactory matrixFactory;
	
	public DummyDirectedAxonsComponentFactory(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}
	
	@Override
	public DirectedAxonsComponent<Neurons, Neurons> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return new DummyDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			Matrix connectionWeights, Matrix biases) {
		return new DummyDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			boolean scaleOutputs) {
		return new DummyDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight) {
		return new DummyDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return new DummyBatchNormDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		return new DummyBatchNormDirectedAxonsComponent<>(matrixFactory, new DummyAxons<>(matrixFactory, leftNeurons, rightNeurons));
	}

}
