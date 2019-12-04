package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.DefaultFloatArrayFactory;
import org.ml4j.jblas.FloatArrayFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class PoolingFormatterImpl3 implements ConvolutionalFormatter {
	
	private final int filterSize;
	private final int inputDepth;
	private final int inputWidth;
	private final int inputHeight;
	private final int outputWidth;
	private final int outputHeight;
	private final int windowWidth;
	private final int windowHeight;
	private final int paddingHeight;
	private final int paddingWidth;
	private final int windowWidthPositions;
	private final int windowHeightPositions;
	private final int inputHeightWithPadding;
	private final int inputWidthWithPadding;
	private final int examples;
	private final int outputDepth;
	private final int filterWidth;
	private final int filterHeight;
	private final int strideWidth;
	private final int strideHeight;
	private final int windowSpanWidth;
	private final int windowSpanHeight;
	private final Neurons3D leftNeurons;
	private final Neurons3D rightNeurons;
	private final FloatArrayFactory floatArrayFactory;
	private final boolean n;
	
	public PoolingFormatterImpl3(Neurons3D leftNeurons, Neurons3D rightNeurons, int strideWidth, int strideHeight, int paddingWidth, int paddingHeight,  int examples, boolean n) {
		this.inputWidth = leftNeurons.getWidth();
		this.inputDepth = leftNeurons.getDepth();
		this.inputHeight = leftNeurons.getHeight();
		this.outputWidth = rightNeurons.getWidth();
		this.outputHeight = rightNeurons.getHeight();
		this.outputDepth = rightNeurons.getDepth();
		this.examples = examples;
		this.strideWidth = strideWidth;
		this.strideHeight = strideHeight;
		this.paddingWidth = paddingWidth;
		this.paddingHeight =paddingHeight;
		this.inputWidthWithPadding = inputWidth + paddingWidth * 2;
		this.inputHeightWithPadding = inputHeight + paddingHeight * 2;
		this.filterWidth = inputWidthWithPadding + (1 - outputWidth) * (strideWidth);
		this.filterHeight = inputHeightWithPadding + (1 - outputHeight) * (strideHeight);
		this.windowWidthPositions = filterWidth;
		this.windowHeightPositions = filterHeight;
		
		this.windowSpanWidth = inputWidthWithPadding - filterWidth + 1;
		this.windowSpanHeight = inputHeightWithPadding - filterHeight + 1;
		this.windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1)/strideWidth;
		this.windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1)/strideHeight;
		
		this.filterSize = filterWidth * filterHeight;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.floatArrayFactory = new DefaultFloatArrayFactory();
		this.n = n;
	}

	@Override
	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation activations) {
		return activations.asImageNeuronsActivation(leftNeurons).im2Col2(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth);
	}
	
	private Matrix getAllRestructured(MatrixFactory matrixFactory, Matrix matrix) {
		float[] matrixData = matrix.getRowByRowArray();
		float[] restructuredMatrixData = floatArrayFactory.createFloatArray(filterSize * inputDepth * outputWidth * outputHeight * examples);
		int index =0;

		for (int sr = 0; sr < windowHeightPositions; sr++ ) {
			for (int sc = 0; sc < windowWidthPositions; sc++) {
				for(int d = 0 ; d < inputDepth; d++) {

					int restructuredMatrixRow = index;
					//int restructuredMatrixStartPosition  =  restructuredMatrixRow * inputDepth * outputWidth * outputHeight * examples;
					int restructuredMatrixStartPosition = restructuredMatrixRow * inputDepth *outputWidth * outputHeight * examples
					+ d * outputWidth * outputHeight * examples;
					populateSingleRestructuredRowForDepth(matrixData, restructuredMatrixData, d, sr, sc, examples, restructuredMatrixStartPosition);
				}				
			index++;

		}
		}
		Matrix m =  matrixFactory.createMatrixFromRowsByRowsArray(filterSize, inputDepth * outputWidth * outputHeight * examples, restructuredMatrixData);
		return m;

	}

	private void populateSingleRestructuredRowForDepth(float[] matrixData, float[] restructured , int depth, int sr, int sc, int examples, int restructuredMatrixStartPosition) {
		int startIndex = restructuredMatrixStartPosition;

		for (int r = 0; r < windowHeight; r++) {
			int actualRow = r * strideHeight + sr;
		
			if (actualRow >= paddingHeight && actualRow < (inputHeightWithPadding - paddingHeight)) {
				if (false && strideWidth == 1) {
					int msc = Math.max(sc, paddingWidth);
					int actualRowWithoutPadding = actualRow - paddingHeight;
					int actualColumnWithoutPadding = (msc - paddingWidth);
					int start = (actualRow * inputWidthWithPadding + msc) * examples;
					int startWithoutPadding = (actualRowWithoutPadding * inputWidth + actualColumnWithoutPadding)
							* examples;
					int end = (actualRow * inputWidthWithPadding
							+ Math.min(sc + windowWidth, inputWidthWithPadding - paddingWidth)) * examples;
					int endWithoutPadding = (actualRowWithoutPadding * 1) * inputWidth * examples;

					int length = end - start;
					// int lengthWithoutPadding = endWithoutPadding - startWithoutPadding;
					// int length = examples;
					// int shift = 0;
					// int shift = start - (actualRow * inputWidthWithPadding + sc ) * examples;
					int shift = start - (actualRow * inputWidthWithPadding + sc) * examples;
					// TODO ML Changed for pooling
					System.arraycopy(matrixData, startWithoutPadding + depth * inputWidth * inputHeight * examples,
							restructured, startIndex + shift, length);
				} else {
					for (int c = 0; c < windowWidth; c++) {
						// int msc = Math.max(sc, paddingWidth);
						int actualColumn = sc + c * strideWidth;
						if (actualColumn >= paddingWidth && actualColumn < (inputWidthWithPadding - paddingWidth)) {
							int actualRowWithoutPadding = actualRow - paddingHeight;
							int actualColumnWithoutPadding = (actualColumn - paddingWidth);
							int start = (actualRow * inputWidthWithPadding + actualColumn) * examples;
							int startWithoutPadding = (actualRowWithoutPadding * inputWidth
									+ actualColumnWithoutPadding) * examples;
							int end = (actualRow * inputWidthWithPadding
									+ Math.min(sc + windowWidth, inputWidthWithPadding - paddingWidth)) * examples;
							int endWithoutPadding = (actualRowWithoutPadding * 1) * inputWidth * examples;

							int length = end - start;
							// int lengthWithoutPadding = endWithoutPadding - startWithoutPadding;
							// int length = examples;
							// int shift = 0;
							// int shift = start - (actualRow * inputWidthWithPadding + sc ) * examples;
							int shift = 0;
							// TODO ML Changed for pooling
							System.arraycopy(matrixData,
									startWithoutPadding + depth * inputWidth * inputHeight * examples, restructured,
									startIndex + shift, examples);
						}
						startIndex = startIndex + examples;
					}
				}
			} 
			else {
				startIndex = startIndex + examples * windowWidth;
			}
			if (false && strideWidth == 1) {
				startIndex = startIndex + windowWidth * examples;
			}
		} 
		
}
	
	private void reverseSingleRestructuredRowForDepth(float[] matrixDataToPopulate, int[] countsToPopulate, Matrix matrix, int depth, int sr, int sc, int examples, int startPosition) {
		int startIndex = startPosition;
		float[] matrixData = matrix.getRowByRowArray();
		for (int r = 0; r < windowHeight; r++) {
			int actualRow = r * strideHeight + sr;
			// Ignore padding rows
			if (actualRow >= paddingHeight && actualRow < (inputHeightWithPadding - paddingHeight)) {
						for (int c = 0; c < windowWidth; c++) {
							int actualColumn = sc + c * strideWidth;
							if (actualColumn >= paddingWidth && actualColumn < (inputWidthWithPadding - paddingWidth)) {

								int actualRowWithoutPadding = actualRow - paddingHeight;
								int actualColumnWithoutPadding = actualColumn - paddingWidth;
								int matrixDataToPopulateStartWithoutPadding = (actualRowWithoutPadding * inputWidth
										+ (actualColumnWithoutPadding)) * examples;
								int matrixDataToPopulateEndWithoutPadding = matrixDataToPopulateStartWithoutPadding
										+ examples;

								// int matrixDataToPopulateEnd = (actualRow * inputWidthWithPadding +
								// Math.min(sc + windowWidth, inputWidthWithPadding - paddingWidth)) * examples;
								// int matrixDataToPopulateEndWithoutPadding = (actualRowWithoutPadding *
								// inputWidth + Math.min(sc + windowWidth - paddingWidth, inputWidthWithPadding
								// - paddingWidth - paddingWidth)) * examples;

								int shiftRight = 0;
								// int shiftRight = matrixDataToPopulateStart - (actualRow *
								// inputWidthWithPadding + sc) * examples;
								// int shiftRightWithoutPadding = matrixDataToPopulateStartWithoutPadding -
								// (actualRowWithoutPadding * inputWidth + (msc - paddingWidth)) * examples;

								int restructuredDataStart = startIndex;
								// float[] subArray = getSubArray(matrixData, restructuredDataStart,
								// restructuredDataEnd);
								// addSubArray(matrixDataToPopulate, countsToPopulate, subArray,
								// matrixDataStart);
								addSubArray2(matrixDataToPopulate, countsToPopulate, matrixData,
										restructuredDataStart + shiftRight,
										matrixDataToPopulateStartWithoutPadding
												+ depth * inputWidth * inputHeight * examples,
										matrixDataToPopulateEndWithoutPadding
												+ depth * inputWidth * inputHeight * examples);
							}
							startIndex = startIndex + examples;
						}
			}
		}
	}

	@Override
	public Matrix reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix initialOutputMatrix) {
		// Reverse restructure into target matrix, with counts, should we wish to average
		float[] matrixDataToPopulate = floatArrayFactory.createFloatArray(leftNeurons.getNeuronCountExcludingBias() * examples);
		//int[] countsToPopulate = new int[leftNeurons.getNeuronCountExcludingBias() * examples];
		reverseAllRestructured(matrixFactory, matrixDataToPopulate, null, initialOutputMatrix, examples);
		return matrixFactory.createMatrixFromRowsByRowsArray(leftNeurons.getNeuronCountExcludingBias(), examples, matrixDataToPopulate);	
	}
	
	private void reverseAllRestructured(MatrixFactory matrixFactory, float[] matrixDataToPopulate, int[] countsToPopulate, Matrix reverseRestructured, int examples) {
		int ind = 0;
		for (int sr = 0; sr < windowWidthPositions; sr++) {
			for (int sc = 0; sc < windowHeightPositions; sc++) {
			
				/*
				Matrix singleRestructuredRowForEachDepth = matrixFactory.createMatrix(inputDepth, windowWidth * windowHeight *  examples);
				System.out.println(singleRestructuredRowForEachDepth.getRows() + ":" + singleRestructuredRowForEachDepth.getColumns());
				for (int d = 0; d < inputDepth; d++) {
					int reverseRestructuredRow = d * windowWidthPositions * windowHeightPositions + ind;

					Matrix r = reverseRestructured.getRow(reverseRestructuredRow);
					System.out.println(r.getRows() + ":" + r.getColumns());
					singleRestructuredRowForEachDepth.putRow(d, r);
				}
				*/
				
				for (int d = 0; d < inputDepth; d++) {
					int reverseRestructuredRow = ind;
					int restructuredMatrixStartPosition = reverseRestructuredRow *  inputDepth * outputWidth * outputHeight * examples 
							 + d * outputWidth * outputHeight * examples;
					reverseSingleRestructuredRowForDepth(matrixDataToPopulate, countsToPopulate, reverseRestructured, d, sr, sc, examples, restructuredMatrixStartPosition);
				}
				
				//reverseSingleRestructuredRowForEachDepth(matrixDataToPopulate, countsToPopulate, singleRestructuredRowForEachDepth, sr, sc, examples);
				ind++;
			}
		}
	}
	
	private void reverseSingleRestructuredRowForEachDepth(float[] matrixDataToPopulate, int[] countsToPopulate, Matrix matrix, int sr, int sc, int examples) {
		int startIndex = 0;
		float[] matrixData = matrix.getRowByRowArray();
		for (int d = 0; d < inputDepth; d++) {
			for (int r = 0; r < windowHeight; r++) {
				int actualRow = r  + sr;
				// Ignore padding rows
				if (actualRow >= paddingHeight && actualRow < (inputHeightWithPadding - paddingHeight)) {
					int msc = Math.max(sc, paddingWidth);
					int actualRowWithoutPadding = actualRow - paddingHeight;
	
					int matrixDataToPopulateStart = (actualRow * inputWidthWithPadding + msc) * examples;
					int matrixDataToPopulateStartWithoutPadding = (actualRowWithoutPadding * inputWidth + (msc - paddingWidth)) * examples;
					
					//int matrixDataToPopulateEnd = (actualRow * inputWidthWithPadding  + Math.min(sc + windowWidth, inputWidthWithPadding - paddingWidth)) * examples;
					int matrixDataToPopulateEndWithoutPadding = (actualRowWithoutPadding * inputWidth  + Math.min(sc + windowWidth - paddingWidth, inputWidthWithPadding - paddingWidth - paddingWidth)) * examples;
	
					int shiftRight = matrixDataToPopulateStart - (actualRow * inputWidthWithPadding + sc) * examples;
					//int shiftRightWithoutPadding = matrixDataToPopulateStartWithoutPadding - (actualRowWithoutPadding * inputWidth + (msc - paddingWidth)) * examples;
	
					int restructuredDataStart = r * windowWidth * examples + + d * windowWidth * windowHeight * examples;
					//float[] subArray = getSubArray(matrixData, restructuredDataStart, restructuredDataEnd);
					//addSubArray(matrixDataToPopulate, countsToPopulate, subArray, matrixDataStart);
					addSubArray2(matrixDataToPopulate, countsToPopulate, matrixData, restructuredDataStart + shiftRight, matrixDataToPopulateStartWithoutPadding + d * windowWidth * windowHeight * examples, matrixDataToPopulateEndWithoutPadding + d * windowWidth * windowHeight * examples);
				}
				startIndex = startIndex + windowWidth * examples;
			}
		}
	}
	
	private void addSubArray2(float[] matrixData, int[] counts, float[] data, int dataStart, int matrixDataStart, int matrixDataEnd) {
		for (int i = 0; i < (matrixDataEnd - matrixDataStart); i++) {
			matrixData[i + matrixDataStart] = matrixData[i + matrixDataStart] +  data[i + dataStart];
			if (counts != null) {
				counts[i + matrixDataStart] = counts[i + matrixDataStart] +  1;
			}
		}
	}

	@Override
	public Matrix reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {
		//input.reshape(outputDepth, outputWidth * outputHeight * examples);
		return input.getActivations(matrixFactory);
	}

	@Override
	public Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, Matrix output) {
		output.asEditableMatrix().reshape(outputWidth * outputHeight * outputDepth , examples);
		return output;
	}

	@Override
	public Matrix getIndexes() {
		throw new UnsupportedOperationException();
	}

}
