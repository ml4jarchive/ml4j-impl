package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DummyConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	private int strideWidth;
	private int strideHeight;
	private int paddingWidth;
	private int paddingHeight;
	
	
	public DummyConvolutionalAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, int paddingWidth, int paddingHeight) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.paddingHeight = paddingHeight;
		this.paddingWidth = paddingWidth;
		this.strideHeight = strideHeight;
		this.strideWidth = strideWidth;
	}

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

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
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
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
	public ConvolutionalAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

	@Override
	public int getFilterHeight() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int getFilterWidth() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int getStrideHeight() {
		return strideHeight;
	}

	@Override
	public int getStrideWidth() {
		return strideWidth;
	}

	@Override
	public int getZeroPaddingHeight() {
		return paddingHeight;
	}

	@Override
	public int getZeroPaddingWidth() {
		return paddingWidth;
	}

}
