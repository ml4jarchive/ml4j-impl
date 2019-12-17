package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DummyAxons<L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> implements Axons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	private L leftNeurons;
	private R rightNeurons;

	public DummyAxons(MatrixFactory matrixFactory, L leftNeurons, R rightNeurons) {
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	@Override
	public L getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public R getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		Matrix output = matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), leftNeuronsActivation.getExampleCount());
		NeuronsActivation outputActivation = new NeuronsActivationImpl(output, leftNeuronsActivation.getFeatureOrientation());
		return new DummyAxonsActivation(this, leftNeuronsActivation, outputActivation);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		throw new UnsupportedOperationException();
	}

	@Override
	public A dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

}
