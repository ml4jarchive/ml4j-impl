package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.ConvolutionalFormatter;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyMaxPoolingReformatterImpl implements ConvolutionalFormatter {

	private int examples;
	private int targetInputFeatureCount;
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	public DummyMaxPoolingReformatterImpl(MatrixFactory matrixFactory, int examples, int targetInputFeatureCount,
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		this.examples = examples;
		this.targetInputFeatureCount = targetInputFeatureCount;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	public Matrix reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation activations1) {		
		return  matrixFactory.createMatrix(this.examples * rightNeurons.getNeuronCountExcludingBias(), 1);
	}
	public Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix matrix) {
		return matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), this.examples);
	}

	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation input) {
		Matrix m =  matrixFactory.createMatrix(targetInputFeatureCount, this.examples * rightNeurons.getNeuronCountExcludingBias());
		return m;
	}

	public Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, Matrix output) {
		return matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), this.examples);
	}

	@Override
	public Matrix getIndexes() {
		// TODO Auto-generated method stub
		return null;
	}
}
