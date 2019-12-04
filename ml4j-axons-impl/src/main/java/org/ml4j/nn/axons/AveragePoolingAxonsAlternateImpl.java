package org.ml4j.nn.axons;

import java.util.Date;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AveragePoolingAxonsAlternateImpl extends BasePoolingAxonsImpl<AveragePoolingAxons> implements AveragePoolingAxons {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(AveragePoolingAxonsAlternateImpl.class);
	
	public AveragePoolingAxonsAlternateImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig axonsConfig) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, false);
	}

	@Override
	public AveragePoolingAxons dup() {
		return new AveragePoolingAxonsAlternateImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	protected AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix inputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		LOGGER.debug("Pushing left to right through average pooling axons:" + inputMatrix.getClass().getName());
				
		long start = new Date().getTime();
		
		int inputMatrixRows = inputMatrix.getRows();
		int inputMatrixColumns = inputMatrix.getColumns();
		
		if (!(inputMatrix instanceof ImageNeuronsActivation)) {
			inputMatrix = inputMatrix.asImageNeuronsActivation(leftNeurons);
		}
		ImageNeuronsActivation c = (ImageNeuronsActivation)inputMatrix;

		Matrix inputOnes = axonsContext.getMatrixFactory().createOnes(inputMatrixRows, inputMatrixColumns);
		Matrix reformatted = reformatLeftToRightInput(inputMatrix, axonsContext.getMatrixFactory());
		Matrix counts = reformatLeftToRightInput(new ImageNeuronsActivationImpl(inputOnes, (Neurons3D)c.getNeurons(), 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false), axonsContext.getMatrixFactory());
		EditableMatrix counts2 = counts.columnSums().asEditableMatrix();
		for (int i = 0; i < counts2.getLength(); i++) {
			 if (counts2.get(i) == 0) {
				 counts2.put(i, 1);
			 }
		}
		
		NeuronsActivation input = new NeuronsActivationImpl(reformatted, featureOrientation);
		
		Matrix origOutput = reformatted.columnSums().asEditableMatrix().diviRowVector(counts2);
		
		
		
		
			Matrix leftToRightOutput = reformatLeftToRightOutput(axonsContext.getMatrixFactory(), inputMatrixColumns, origOutput);
			
			long end = new Date().getTime();
			long t = end - start;
			Timings.addTime(TimingKey.LEFT_TO_RIGHT_AVG, t);
			
			LOGGER.debug("End Pushing left to right through average pooling axons");
			//reformatted.close();
			return new AxonsActivationImpl(this, null, input, new ImageNeuronsActivationImpl(leftToRightOutput, rightNeurons, featureOrientation, false), new Neurons(rightNeurons.getNeuronCountExcludingBias() * inputMatrixRows, false), rightNeurons, false);	
		//}

		
	}
	
	

	@Override
	protected AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix inputDropoutMask,
			NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
		LOGGER.debug("Pushing right to left through average pooling axons:" + inputMatrix.getClass().getName());
		long start = new Date().getTime();
		try (InterrimMatrix reformattedRow = reformatRightToLeftInput(inputMatrix, axonsContext.getMatrixFactory())
				.asInterrimMatrix()) {

			int filterWidth = leftNeurons.getWidth() + (2 * getPaddingWidth()) + (1 - rightNeurons.getWidth()) * (getStrideWidth());
			int filterHeight = leftNeurons.getHeight() + (2 * getPaddingHeight()) + (1 - rightNeurons.getHeight()) * (getStrideHeight());
			int filterElementCount = filterWidth * filterHeight;
			try (InterrimMatrix reformattedAvgRow = reformattedRow.div(filterElementCount).asInterrimMatrix()) {
				try (InterrimMatrix reformattedAvgMatrix = axonsContext.getMatrixFactory().createMatrix(filterElementCount,
						reformattedAvgRow.getColumns() * reformattedAvgRow.getRows()).asInterrimMatrix() ) {
				for (int r = 0; r < reformattedAvgMatrix.getRows(); r++) {
					reformattedAvgMatrix.asEditableMatrix().putRow(r, reformattedAvgRow);
				}
				LOGGER.debug("Output average pooling axons:" + reformattedAvgMatrix.getRows() + ":" + reformattedAvgMatrix.getColumns());

				int reformattedAvgMatrixRows = reformattedAvgMatrix.getRows();
				NeuronsActivation input = new NeuronsActivationImpl(reformattedAvgMatrix, featureOrientation);

				Matrix reformattedOutputMatrix = reformatRightToLeftOutput(axonsContext, inputMatrix.getExampleCount(),
						reformattedAvgMatrix);
				
				LOGGER.debug("End Pushing right to left through average pooling axons");

				long end = new Date().getTime();
				long t = end - start;
				Timings.addTime(TimingKey.RIGHT_TO_LEFT_AVG, t);

				return new AxonsActivationImpl(this, null, input,
						new ImageNeuronsActivationImpl(reformattedOutputMatrix, leftNeurons, featureOrientation, false), leftNeurons,
						new Neurons(reformattedAvgMatrixRows, false), true);
				}
			}

		}

	}
}
