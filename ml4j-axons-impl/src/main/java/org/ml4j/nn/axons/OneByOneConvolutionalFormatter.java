package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class OneByOneConvolutionalFormatter implements ConvolutionalFormatter {
	
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	private int examples;
	
	public OneByOneConvolutionalFormatter(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, int paddingWidth, int paddingHeight,  int examples) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.examples = examples;
	}
	

	@Override
	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation activations) {
		EditableMatrix out = activations.getActivations(matrixFactory).dup().asEditableMatrix();
		out.reshape(leftNeurons.getDepth(), leftNeurons.getWidth() * leftNeurons.getHeight() * examples);
		return out;
	}

	@Override
	public Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix output) {
		EditableMatrix out = output.dup().asEditableMatrix();
		out.reshape(leftNeurons.getNeuronCountExcludingBias(), examples);
		return out;
	}

	@Override
	public Matrix reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation activations) {
		EditableMatrix m = activations.getActivations(matrixFactory).asEditableMatrix();
		m.reshape(rightNeurons.getDepth(), rightNeurons.getWidth() * rightNeurons.getHeight() * examples);
		return m;
	}

	@Override
	public Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, Matrix output) {
		output.asEditableMatrix().reshape(rightNeurons.getNeuronCountExcludingBias(), examples);
		return output;
	}

	@Override
	public Matrix getIndexes() {
		// TODO Auto-generated method stub
		return null;
	}

}
