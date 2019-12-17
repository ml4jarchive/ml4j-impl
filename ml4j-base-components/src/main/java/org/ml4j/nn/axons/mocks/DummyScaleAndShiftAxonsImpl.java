package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DummyScaleAndShiftAxonsImpl<N extends Neurons> implements ScaleAndShiftAxons<N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	public DummyScaleAndShiftAxonsImpl(MatrixFactory matrixFactory, N leftNeurons, N rightNeurons) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	private N leftNeurons;
	private N rightNeurons;

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
		return null;
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
	public N getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public N getRightNeurons() {
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
	public ScaleAndShiftAxons<N> dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

	@Override
	public Matrix getScaleColumnVector() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getShiftColumnVector() {
		throw new UnsupportedOperationException();
	}

}
