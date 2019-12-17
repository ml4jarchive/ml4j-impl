package org.ml4j.nn.axons.mocks;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DummyMaxPoolingAxonsImpl implements MaxPoolingAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	private int strideWidth;
	private int strideHeight;
	
	
	public DummyMaxPoolingAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.strideHeight = strideHeight;
		this.strideWidth = strideWidth;
	}

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

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
	public MaxPoolingAxons dup() {
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
}
