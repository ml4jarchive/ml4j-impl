package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.ConvolutionalFormatter;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyConvolutionalReformatterImpl implements ConvolutionalFormatter {

	private int examples;
	private int preTransformOutputFeatureCount;
	private int outputChannels;
	private int targetInputFeatureCount;
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	public DummyConvolutionalReformatterImpl(MatrixFactory matrixFactory, int examples, int targetInputFeatureCount,
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		this.examples = examples;
		this.preTransformOutputFeatureCount = rightNeurons.getHeight() * rightNeurons.getWidth();
		this.outputChannels = rightNeurons.getDepth();
		this.targetInputFeatureCount = targetInputFeatureCount;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	public Matrix reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation activations1) {		
		return  matrixFactory.createMatrix(outputChannels, this.preTransformOutputFeatureCount * activations1.getExampleCount());
	}
	public Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix matrix, int[] targetIndexes) {
		return matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), this.examples);
	}

	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, Matrix input, boolean maxPooling) {
		return matrixFactory.createMatrix(targetInputFeatureCount, this.examples * preTransformOutputFeatureCount);
	}

	public Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, Matrix output) {
		return matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), this.examples);
	}

	@Override
	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation activations) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix initialOutputMatrix) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix getIndexes() {
		// TODO Auto-generated method stub
		return null;
	}
}
