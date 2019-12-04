package org.ml4j.nn.axons;

import java.io.Serializable;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class Axons3DBase<A extends Axons<Neurons3D, Neurons3D, A>>
		extends AxonsBaseAlternateImpl<Neurons3D, Neurons3D, A, Axons3DConfig> implements Serializable {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(Axons3DBase.class);

	protected boolean independentInputOutputChannels;
	protected MatrixFactory matrixFactory;

	public Axons3DBase(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig axonsConfig, boolean independentInputOutputChannels) {
		super(leftNeurons, rightNeurons, axonsConfig);
		this.matrixFactory = matrixFactory;
		this.independentInputOutputChannels = independentInputOutputChannels;
	}
	
	protected abstract ConvolutionalFormatter createConvolutionalFormatter(int examples);


	public int getStrideWidth() {
		return config.getStrideWidth();
	}

	public int getStrideHeight() {
		return config.getStrideHeight();
	}

	public int getPaddingWidth() {
		return config.getPaddingWidth();
	}

	public int getPaddingHeight() {
		return config.getPaddingHeight();
	}

	public int getFilterWidth() {
		return config.getFilterWidth();
	}

	public int getFilterHeight() {
		return config.getFilterHeight();
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}
	
	protected Matrix reformatRightToLeftInput(NeuronsActivation activations, MatrixFactory matrixFactory) {

		LOGGER.debug("Reformatting right to left input:" + activations.getFeatureCount() + ":" + activations.getExampleCount());
	
		ConvolutionalFormatter formatter = createConvolutionalFormatter(activations.getExampleCount());
		Matrix reformatted = formatter.reformatRightToLeftInput(matrixFactory, activations);

		LOGGER.debug("End Reformatting right to left input:" + reformatted.getRows() + ":" + reformatted.getColumns());

		return activations.getActivations(matrixFactory);

	}
	
	protected Matrix reformatLeftToRightInput(NeuronsActivation activations, MatrixFactory matrixFactory) {
		LOGGER.debug("Reformatting left to right input: " + activations.getFeatureCount() + ";" + activations.getExampleCount() + ":" + activations.getClass());
		
		ConvolutionalFormatter formatter = createConvolutionalFormatter(activations.getExampleCount());
		Matrix reformatted = formatter.reformatLeftToRightInput(matrixFactory, activations);
		LOGGER.debug("End Reformatting left to right input: " + reformatted.getRows() + ";" + reformatted.getColumns());
		return reformatted;
	}
	
	

	protected Matrix reformatRightToLeftOutput(AxonsContext axonsContext, int inputExamples,
			Matrix initialOutput) {

		LOGGER.debug("Reformatting right to left output:" + initialOutput.getRows() + ":"
				+ initialOutput.getColumns() + ":" + "with example count :" + inputExamples);

		ConvolutionalFormatter formatter = createConvolutionalFormatter(inputExamples);

		Matrix reformatted = formatter.reformatRightToLeftOutput(matrixFactory, initialOutput);
		LOGGER.debug("End Reformatting right to left output:" + reformatted.getRows() + ":" + reformatted.getColumns());
		return reformatted;

	}

	protected Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, int inputExamples,
			Matrix origOutput) {

		LOGGER.debug("Reformatting left to right output:" + origOutput.getRows() + ":" + origOutput.getColumns());

		ConvolutionalFormatter formatter = createConvolutionalFormatter(inputExamples);

		Matrix reform = formatter.reformatLeftToRightOutput(matrixFactory, origOutput);

		LOGGER.debug("End Reformatting left to right output:" + reform.getRows() + ":" + reform.getColumns());

		return reform;
	}

}
