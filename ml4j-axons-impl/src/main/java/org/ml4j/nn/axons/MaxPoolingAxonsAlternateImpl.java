package org.ml4j.nn.axons;

import java.util.Date;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MaxPoolingAxonsAlternateImpl extends BasePoolingAxonsImpl<MaxPoolingAxons> implements MaxPoolingAxons {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(MaxPoolingAxonsAlternateImpl.class);

	private boolean scaleOutputs;
	
	public MaxPoolingAxonsAlternateImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig axonsConfig, boolean scaleOutputs) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, false);
		this.scaleOutputs = scaleOutputs;
	}
	
	@Override
	protected Matrix reformatLeftToRightInput(NeuronsActivation activations, MatrixFactory matrixFactory) {
		
		LOGGER.debug("Formatting left to right max pooling axons");
		
		EditableMatrix reformatted = super.reformatLeftToRightInput(activations, matrixFactory).asEditableMatrix();
		if (scaleOutputs) {
	    	   int outputDim =
	   		        (int) (this.getRightNeurons().getNeuronCountIncludingBias()
	   		            / this.getRightNeurons().getDepth());
	   		    int inputDim = (int) (
	   		        this.getLeftNeurons().getNeuronCountIncludingBias() / getLeftNeurons().getDepth());
	   		    
	   		    float scaleDown = inputDim / outputDim;
	    	reformatted.muli(scaleDown);
	    }
		
		LOGGER.debug("End formatting left to right max pooling axons");

		return reformatted;
	}

	@Override
	public MaxPoolingAxons dup() {
		return new MaxPoolingAxonsAlternateImpl(matrixFactory, leftNeurons, rightNeurons, config, scaleOutputs);
	}

	@Override
	protected AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix inputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		
		LOGGER.debug("Pushing left to right max pooling axons:" + inputMatrix.getFeatureCount() + ":" + inputMatrix.getExampleCount());

		long start = new Date().getTime();
		
		int inputMatrixColumns =  inputMatrix.getExampleCount();
		
		if (featureOrientation != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalStateException("Currently only ROWS_SPAN_FEATURE_SET orientation supported");
		}
		
		Matrix reformatted = reformatLeftToRightInput(inputMatrix, axonsContext.getMatrixFactory());
		EditableMatrix maxes = (EditableMatrix)axonsContext.getMatrixFactory().createMatrix(reformatted.getRows(), reformatted.getColumns());			
		
		EditableMatrix origOutput = (EditableMatrix)axonsContext.getMatrixFactory().createMatrix(1, reformatted.getColumns());
		int[] maxInts = reformatted.columnArgmaxs();
		for (int c = 0; c < reformatted.getColumns(); c++) {
			if (maxInts[c] != -1) { 
				maxes.put(maxInts[c], c,  1);
				origOutput.put(0, c, reformatted.get(maxInts[c], c));
			} 
		}

		Matrix leftToRightOutput = reformatLeftToRightOutput(axonsContext.getMatrixFactory(), inputMatrixColumns, origOutput);
		NeuronsActivation input = new NeuronsActivationImpl(reformatted, featureOrientation);
		
		LOGGER.debug("End Pushing left to right max pooling axons:" + leftToRightOutput.getRows() + ":" + leftToRightOutput.getColumns());
		long end = new Date().getTime();
		long t = end - start;
		Timings.addTime(TimingKey.LEFT_TO_RIGHT_MAX, t);
		
		//reformatted.close();
		return new AxonsActivationImpl(this, maxes, input, new ImageNeuronsActivationImpl(leftToRightOutput, rightNeurons, featureOrientation, false), new Neurons(rightNeurons.getNeuronCountExcludingBias() * inputMatrixColumns, false), rightNeurons, false);
	}

	@Override
	protected AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix previousLeftToRightInputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		
		LOGGER.debug("Pushing right to left max pooling axons:" + inputMatrix.getFeatureCount() + ":" + inputMatrix.getExampleCount());
		long start = new Date().getTime();
		int examples = inputMatrix.getExampleCount();

		Matrix reformatted = reformatRightToLeftInput(inputMatrix, axonsContext.getMatrixFactory());

		LOGGER.debug("Reformatted max pooling axons:" + reformatted.getRows() + ":" + reformatted.getColumns());

	  	try (InterrimMatrix outputMatrix = previousLeftToRightInputDropoutMask.asInterrimMatrix()) {
		LOGGER.debug("Output max pooling axons:" + outputMatrix.getRows() + ":" + outputMatrix.getColumns());

		EditableMatrix reformatted2 = (EditableMatrix)axonsContext.getMatrixFactory().createMatrix(outputMatrix.getRows(), outputMatrix.getColumns());
		for (int r = 0; r < reformatted2.getRows(); r++) {
			reformatted2.putRow(r, reformatted.asEditableMatrix());
		}
		//reformatted.close();
		outputMatrix.asEditableMatrix().muli(reformatted2);
		
		NeuronsActivation input = new NeuronsActivationImpl(reformatted2, featureOrientation);
		Matrix reformattedOutputMatrix = reformatRightToLeftOutput(axonsContext, examples, outputMatrix);

		LOGGER.debug("End Pushing right to left max pooling axons");
		long end = new Date().getTime();
		long t = end - start;
		Timings.addTime(TimingKey.RIGHT_TO_LEFT_MAX, t);
		int reformattedRows = reformatted2.getRows();
		reformatted2.close();
		return new AxonsActivationImpl(this, null, input, new ImageNeuronsActivationImpl(reformattedOutputMatrix, leftNeurons, featureOrientation, false), leftNeurons, new Neurons(reformattedRows, false), true);	
	  	}
	}
}
