package org.ml4j.nn.components.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsFactory;
import org.ml4j.nn.axons.AxonsFactoryImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class DirectedAxonsComponentFactoryImpl implements DirectedAxonsComponentFactory {

	private AxonsFactory axonsFactory;
	
	public DirectedAxonsComponentFactoryImpl(AxonsFactory axonsFactory) {
		this.axonsFactory = axonsFactory;
	}
	
	public DirectedAxonsComponentFactoryImpl(MatrixFactory matrixFactory) {
		this.axonsFactory = new AxonsFactoryImpl(matrixFactory);
	}
	
	@Override
	public DirectedAxonsComponent<Neurons, Neurons> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createFullyConnectedAxons(leftNeurons, rightNeurons, connectionWeights, biases));
	}
	
	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight, Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createConvolutionalAxons(leftNeurons, rightNeurons, strideWidth, strideHeight, paddingWidth, paddingHeight,  connectionWeights, biases));
	}
	
	protected <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponent(Axons<L, R, ?> axons) {
		return new DirectedAxonsComponentImpl<>(axons);
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight, boolean scaleOutputs) {
		return createDirectedAxonsComponent(axonsFactory.createMaxPoolingAxons(leftNeurons, rightNeurons, scaleOutputs, strideWidth, strideHeight, paddingWidth, paddingHeight));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight) {
		return createDirectedAxonsComponent(axonsFactory.createAveragePoolingAxons(leftNeurons, rightNeurons, strideWidth, strideHeight, paddingWidth, paddingHeight));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(
			N leftNeurons, N rightNeurons) {
		return new BatchNormDirectedAxonsComponentImpl<>(axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, null, null));

	}
	
	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(
			N leftNeurons, N rightNeurons, Matrix gamma, Matrix beta, Matrix means, Matrix stdev) {
		return new BatchNormDirectedAxonsComponentImpl<>(axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, gamma, beta), means, stdev);
	}
}
