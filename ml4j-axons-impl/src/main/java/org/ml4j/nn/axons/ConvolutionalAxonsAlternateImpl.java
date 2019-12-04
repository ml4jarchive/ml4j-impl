package org.ml4j.nn.axons;

import java.util.Date;

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

public class ConvolutionalAxonsAlternateImpl extends Axons3DBase<ConvolutionalAxons> implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(ConvolutionalAxonsAlternateImpl.class);
	private FullyConnectedAxons fullyConnectedAxons;

	private FullyConnectedAxonsFactory fullyConnectedAxonsFactory;	
	private boolean reversed;
	
	public ConvolutionalAxonsAlternateImpl(FullyConnectedAxonsFactory fullyConnectedAxonsFactory,
			MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig axonsConfig, boolean reversed) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, true);
		this.matrixFactory = matrixFactory;
		this.fullyConnectedAxonsFactory = fullyConnectedAxonsFactory;
		this.fullyConnectedAxons = createFullyConnectedAxons(fullyConnectedAxonsFactory, null, null);
		this.reversed = reversed;
	}
	
	protected ConvolutionalFormatter createConvolutionalFormatter(int examples) {
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();
		
		int inputWidthWithPadding = inputWidth + getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (getStrideHeight());
		if (filterWidth == 1 && filterHeight == 1 && getPaddingWidth() == 0 && getPaddingHeight() == 0 && getStrideHeight() == 1 && getStrideWidth() == 1) {
			return new OneByOneConvolutionalFormatter(leftNeurons, rightNeurons, getStrideWidth(), getStrideHeight(), getPaddingWidth(), getPaddingHeight(), examples);	
		} else {
			return new ConvolutionalFormatterImpl3(leftNeurons, rightNeurons, getStrideWidth(), getStrideHeight(), getPaddingWidth(), getPaddingHeight(), examples);	
		}
	}

	public ConvolutionalAxonsAlternateImpl(FullyConnectedAxonsFactory fullyConnectedAxonsFactory,
			MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig axonsConfig,
			Matrix connectionWeights, Matrix biases, boolean reversed) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, true);
		this.matrixFactory = matrixFactory;
		this.fullyConnectedAxonsFactory = fullyConnectedAxonsFactory;
		this.fullyConnectedAxons = createFullyConnectedAxons(fullyConnectedAxonsFactory,connectionWeights, biases);
		this.reversed = reversed;
	}

	private FullyConnectedAxons createFullyConnectedAxons(FullyConnectedAxonsFactory fullyConnectedAxonsFactory, Matrix connectionWeights, Matrix biases) {
		if (fullyConnectedAxons == null) {
			int filterWidth = leftNeurons.getWidth() + (2 * getPaddingWidth()) + (1 - rightNeurons.getWidth()) * (getStrideWidth());
			int filterHeight = leftNeurons.getHeight() + (2 * getPaddingHeight()) + (1 - rightNeurons.getHeight()) * (getStrideHeight());
			int filterSize = filterWidth * filterHeight * leftNeurons.getDepth();
			if (connectionWeights == null) {
				
				this.fullyConnectedAxons = fullyConnectedAxonsFactory
						.createFullyConnectedAxons(
								new Neurons3D(filterWidth, filterHeight, leftNeurons.getDepth(),
										leftNeurons.hasBiasUnit()),
								new Neurons(rightNeurons.getDepth(), false), null, null);

			} else {
				this.fullyConnectedAxons = fullyConnectedAxonsFactory.createFullyConnectedAxons(
						new Neurons3D(filterWidth, filterHeight, leftNeurons.getDepth(),
								leftNeurons.hasBiasUnit()),
						new Neurons3D(1, 1, rightNeurons.getDepth(), false), connectionWeights, biases);
			}
		}
		return fullyConnectedAxons;
	}

	@Override
	public ConvolutionalAxons dup() {
		return new ConvolutionalAxonsAlternateImpl(fullyConnectedAxonsFactory, matrixFactory, leftNeurons, rightNeurons,
				config, reversed);
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public void adjustConnectionWeights(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		fullyConnectedAxons.adjustConnectionWeights(adjustment, adjustmentDirection);
	}

	@Override
	public void adjustLeftToRightBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		fullyConnectedAxons.adjustLeftToRightBiases(adjustment, adjustmentDirection);
	}

	@Override
	public Matrix getDetachedConnectionWeights() {
		return fullyConnectedAxons.getDetachedConnectionWeights();
	}

	@Override
	public Matrix getDetachedLeftToRightBiases() {
		return fullyConnectedAxons.getDetachedLeftToRightBiases();
	}

	@Override
	public Matrix getDetachedRightToLeftBiases() {
		return fullyConnectedAxons.getDetachedRightToLeftBiases();
	}

	@Override
	public int getZeroPaddingWidth() {
		return config.getPaddingWidth();
	}

	@Override
	public int getZeroPaddingHeight() {
		return config.getPaddingHeight();
	}

	@Override
	public void adjustRightToLeftBiases(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		fullyConnectedAxons.adjustRightToLeftBiases(adjustments, adjustmentDirection);
	}

	@Override
	protected AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix inputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		LOGGER.debug("Pushing left to right through Conv axons:" + inputMatrix.getFeatureCount() + ":"
				+ inputMatrix.getExampleCount() + ":" + featureOrientation);
		long start = new Date().getTime();
			
		int inputMatrixColumns = inputMatrix.getExampleCount();
		Matrix reformatted = reformatLeftToRightInput(inputMatrix, axonsContext.getMatrixFactory());
		
		LOGGER.debug("Pushing left to right through FC axons");

		long start2 = new Date().getTime();
		
		AxonsActivation nestedActivation = fullyConnectedAxons
				.pushLeftToRight(new NeuronsActivationImpl(reformatted, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET), null, axonsContext);
		
		
		long end2 = new Date().getTime();

		long t2 = end2 - start2;
		Timings.addTime(TimingKey.LEFT_TO_RIGHT_FC, t2);

		LOGGER.debug("End left to right pushing through FC axons");
		//System.out.println(nestedActivation.getOutput().getStackTrace());
		Matrix leftToRightOutput = reformatLeftToRightOutput(axonsContext.getMatrixFactory(), inputMatrixColumns,
				nestedActivation.getOutput().getActivations(axonsContext.getMatrixFactory()));

		LOGGER.debug("End Pushing left to right through Conv axons");

		long end = new Date().getTime();
		long t = end - start;
		Timings.addTime(TimingKey.LEFT_TO_RIGHT_CONV, t);

		LOGGER.debug("End Pushing left to right through Conv axons:" + leftToRightOutput.getRows() + ":"
				+ leftToRightOutput.getColumns());
			
		/*
		if (nestedActivation.getOutput().getActivations().get == 1) {
			throw new IllegalStateException();
		}
		*/
		
		return new AxonsActivationImpl(this, null, nestedActivation.getPostDropoutInput(),
				new ImageNeuronsActivationImpl(leftToRightOutput, rightNeurons, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false), reversed ? fullyConnectedAxons.getRightNeurons() : fullyConnectedAxons.getLeftNeurons(),
				reversed ? fullyConnectedAxons.getLeftNeurons() : fullyConnectedAxons.getRightNeurons(), false);
	}


	@Override
	protected AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix inputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {

		LOGGER.debug("Pushing right to left through Conv axons");
		long start = new Date().getTime();
		
		int inputMatrixColumns = inputMatrix.getExampleCount();


		Matrix reformatted = reformatRightToLeftInput(inputMatrix, axonsContext.getMatrixFactory());

		LOGGER.debug("Pushing right to left through FC axons");

		long start2 = new Date().getTime();
		
		// TOOD WED
		NeuronsActivation input = new NeuronsActivationImpl(reformatted, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		//NeuronsActivation input = inputMatrix.createDownstreamActivation(reformatted);

		
		AxonsActivation nestedActivation = fullyConnectedAxons
				.pushRightToLeft(input, null, axonsContext);

		long end2 = new Date().getTime();
		long t2 = end2 - start2;
		Timings.addTime(TimingKey.RIGHT_TO_LEFT_FC, t2);

		LOGGER.debug("End pushing right to left through FC axons");


		Matrix outputMatrix = nestedActivation.getOutput().getActivations(axonsContext.getMatrixFactory());
		Matrix reformattedOutputMatrix = reformatRightToLeftOutput(axonsContext, inputMatrixColumns,
				outputMatrix);
		
		//nestedActivation.getPostDropoutInput().close();
		//nestedActivation.getOutput().close();
		if (nestedActivation.getInputDropoutMask() != null) {
			//nestedActivation.getInputDropoutMask().close();
		}
	
		
		LOGGER.debug("End push right to left through Conv axons:" + reformattedOutputMatrix.getRows() + ":"
				+ reformattedOutputMatrix.getColumns());
		long end = new Date().getTime();
		long t = end - start;
		Timings.addTime(TimingKey.RIGHT_TO_LEFT_CONV, t);
		// TODO ML
		return new AxonsActivationImpl(this, null, nestedActivation.getPostDropoutInput(),
				new ImageNeuronsActivationImpl(reformattedOutputMatrix, leftNeurons, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false),
				fullyConnectedAxons.getLeftNeurons(), fullyConnectedAxons.getRightNeurons(), true);
	}

	@Override
	protected boolean isLeftInputDropoutSupported() {
		return true;
	}

}
