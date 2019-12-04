package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

public class ConvolutionalAxonsAlternateLatest extends Axons3DBase<ConvolutionalAxons> implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private ConvolutionalAxonsAlternateImpl leftToRightConvolutionalAxons;
	private ConvolutionalAxonsAlternateImpl rightToLeftConvolutionalAxons;


	public ConvolutionalAxonsAlternateLatest(AxonsFactory axonsFactory, MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig axonsConfig, Matrix kernelMatrix, Matrix biasMatrix, boolean grad) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, true);
		Matrix reversedKernel = grad ? null : getReversedKernel(matrixFactory, leftNeurons, rightNeurons, kernelMatrix, axonsConfig.getPaddingWidth(), axonsConfig.getPaddingHeight(), axonsConfig.getStrideWidth(), axonsConfig.getStrideHeight());
		if (reversedKernel != null) {
			System.out.println("Reversed kernel:" + reversedKernel.getRows() + ":" + reversedKernel.getColumns());
		}
		this.leftToRightConvolutionalAxons = new ConvolutionalAxonsAlternateImpl(axonsFactory,matrixFactory, leftNeurons, rightNeurons, axonsConfig, kernelMatrix, biasMatrix, false);
		this.rightToLeftConvolutionalAxons = new ConvolutionalAxonsAlternateImpl(axonsFactory,matrixFactory, rightNeurons, leftNeurons, axonsConfig,reversedKernel , null, true);
	}
	

	public ConvolutionalAxonsAlternateLatest(AxonsFactory axonsFactory, MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig axonsConfig, boolean grad) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, true);
		this.leftToRightConvolutionalAxons = new ConvolutionalAxonsAlternateImpl(axonsFactory,matrixFactory, leftNeurons, rightNeurons, axonsConfig, false);
		Matrix reversedKernel = grad ? null : getReversedKernel(matrixFactory, leftNeurons, rightNeurons, leftToRightConvolutionalAxons.getDetachedConnectionWeights(), axonsConfig.getPaddingWidth(), axonsConfig.getPaddingHeight(), axonsConfig.getStrideWidth(), axonsConfig.getStrideHeight());
		this.rightToLeftConvolutionalAxons = new ConvolutionalAxonsAlternateImpl(axonsFactory,matrixFactory, new Neurons3D(rightNeurons.getWidth(), rightNeurons.getHeight(), rightNeurons.getDepth(), false), new Neurons3D(leftNeurons.getWidth(), leftNeurons.getHeight(), leftNeurons.getDepth(), false), axonsConfig,reversedKernel , null, true);
	}

	
	public void setGrad(boolean grad) {
		//this.leftToRightConvolutionalAxons.setGrad(grad);
	}
	
	private static Matrix getReversedKernel(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Matrix kernel, int paddingWidth, int paddingHeight, int strideWidth, int strideHeight) {
		float[] data = kernel.getRowByRowArray();
		float[] reversedData = new float[data.length];
		int inputWidthWithPadding = leftNeurons.getWidth() + paddingWidth * 2;
		int inputHeightWithPadding = leftNeurons.getHeight() + paddingHeight * 2;
		int filterWidth = inputWidthWithPadding + (1 - rightNeurons.getWidth()) * (strideWidth);
		int filterHeight = inputHeightWithPadding + (1 - rightNeurons.getHeight()) * (strideHeight);
		//System.out.println(leftNeurons.getDepth() + ":" + leftNeurons.getWidth() + ":" + leftNeurons.getHeight() + ":" + leftNeurons.getWidth());
		//System.out.println(rightNeurons.getDepth() + ":" + rightNeurons.getWidth() + ":" + rightNeurons.getHeight() + ":" + rightNeurons.getWidth());

		for (int o = 0; o < rightNeurons.getDepth(); o++) {
			for (int i = 0; i < leftNeurons.getDepth(); i++) {
				int start1 = o * leftNeurons.getDepth() * filterWidth * filterHeight + i * filterWidth * filterHeight;
				int start2 = i * rightNeurons.getDepth() * filterWidth * filterHeight + o * filterWidth * filterHeight;

				int end1 = start1 + filterHeight * filterWidth - 1;
				int end2 = start2 + filterHeight * filterWidth - 1;

				for (int h = 0; h < filterHeight; h++) {
					for (int w = 0; w < filterWidth; w++) {
						int offset = h * filterWidth + w;
						//System.out.println((start1 + offset) + ":" + (end1 - offset));
						//System.out.println((start2 + offset) + ":" + (end2- offset));

						reversedData[start1 + offset] = data[end2 - offset];
					}
				}
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(leftNeurons.getDepth(), leftNeurons.getWidth() * leftNeurons.getHeight() * rightNeurons.getDepth(), reversedData);
	}

	@Override
	public void adjustConnectionWeights(Matrix adjustment, ConnectionWeightsAdjustmentDirection arg1) {
		leftToRightConvolutionalAxons.adjustConnectionWeights(adjustment, arg1);		
	}

	@Override
	public void adjustLeftToRightBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection arg1) {
		leftToRightConvolutionalAxons.adjustLeftToRightBiases(	adjustment, arg1);

	}

	@Override
	public void adjustRightToLeftBiases(Matrix arg0, ConnectionWeightsAdjustmentDirection arg1) {
		
	}

	@Override
	public Matrix getDetachedConnectionWeights() {
		return leftToRightConvolutionalAxons.getDetachedConnectionWeights();
	}

	@Override
	public Matrix getDetachedLeftToRightBiases() {
		return leftToRightConvolutionalAxons.getDetachedLeftToRightBiases();
	}

	@Override
	public Matrix getDetachedRightToLeftBiases() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ConvolutionalAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext arg0) {
		return !arg0.isWithFreezeOut();
	}

	@Override
	public int getZeroPaddingHeight() {
		return config.getPaddingHeight();
	}

	@Override
	public int getZeroPaddingWidth() {
		return config.getPaddingWidth();
	}

	@Override
	protected ConvolutionalFormatter createConvolutionalFormatter(int examples) {
		throw new UnsupportedOperationException("Unsupported");
	}

	@Override
	protected boolean isLeftInputDropoutSupported() {
		return true;
	}

	@Override
	protected AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix previousRightToLeftInputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {

		AxonsActivation activation =  leftToRightConvolutionalAxons.doPushLeftToRight(inputMatrix, previousRightToLeftInputDropoutMask, featureOrientation, axonsContext);
		
		return new AxonsActivationImpl(this, activation.getInputDropoutMask(), activation.getPostDropoutInput(), activation.getOutput(), activation.getAxons().getLeftNeurons(), activation.getAxons().getRightNeurons(), true);
	}

	@Override
	protected AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix previousLeftToRightInputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		AxonsActivation activation =  rightToLeftConvolutionalAxons.doPushLeftToRight(inputMatrix, previousLeftToRightInputDropoutMask, featureOrientation, axonsContext);
	
		return new AxonsActivationImpl(this, activation.getInputDropoutMask(), activation.getPostDropoutInput(), activation.getOutput(), activation.getAxons().getLeftNeurons(), activation.getAxons().getRightNeurons(), true);

	}

}
