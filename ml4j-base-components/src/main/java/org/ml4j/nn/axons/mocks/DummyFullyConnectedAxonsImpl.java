package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DummyFullyConnectedAxonsImpl implements FullyConnectedAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	public DummyFullyConnectedAxonsImpl(MatrixFactory matrixFactory, Neurons leftNeurons, Neurons rightNeurons) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	private Neurons leftNeurons;
	private Neurons rightNeurons;

	@Override
	public void adjustConnectionWeights(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		// No-op
	}

	@Override
	public void adjustLeftToRightBiases(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		// No-op
	}

	@Override
	public void adjustRightToLeftBiases(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		// No-op
	}

	@Override
	public Matrix getDetachedConnectionWeights() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getDetachedLeftToRightBiases() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getDetachedRightToLeftBiases() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Neurons getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		return new AxonsActivationImpl(this, null, leftNeuronsActivation, new NeuronsActivationImpl(matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), leftNeuronsActivation.getExampleCount()), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET), leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		return new AxonsActivationImpl(this, null, rightNeuronsActivation, new NeuronsActivationImpl(matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), rightNeuronsActivation.getExampleCount()), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET), leftNeurons, rightNeurons);
	}

	@Override
	public FullyConnectedAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

}
