package org.ml4j.nn.datasets.floatarray;

import java.util.stream.Stream;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

import com.codepoetics.protonpack.StreamUtils;

public class FloatArrayDataBatchImpl extends DataBatchImpl<float[]> implements FloatArrayDataBatch {

	private int featureCount;
	private int batchSize;

	public FloatArrayDataBatchImpl(int featureCount, int batchSize) {
		super(batchSize);
		this.batchSize = batchSize;
		this.featureCount = featureCount;
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
	}

	public FloatArrayDataBatchImpl(Stream<float[]> data, int featureCount, int batchSize) {
		super(data, batchSize);
		this.featureCount = featureCount;
		this.batchSize = batchSize;
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		if (batchSize != size()) {
			throw new IllegalStateException();
		}
	}

	public Matrix getAsMatrix(MatrixFactory matrixFactory) {

		Matrix data = matrixFactory.createMatrix(batchSize, featureCount);

		StreamUtils.zipWithIndex(stream())
				.forEach(d -> addToMatrix(matrixFactory, data, d.getValue(), featureCount, (int) d.getIndex()));

		return data.transpose();
	}

	private void addToMatrix(MatrixFactory matrixFactory, Matrix matrix, float[] data, int featureCount, int row) {
		matrix.asEditableMatrix().putRow(row, matrixFactory.createMatrixFromRowsByRowsArray(1, featureCount, data));
	}
	
	
	@Override
	public NeuronsActivation toNeuronsActivation(MatrixFactory matrixFactory) {

		Matrix dataMatrix = matrixFactory.createMatrix(featureCount, batchSize);
		
		Stream<float[]> floatStream = stream();

		StreamUtils.zipWithIndex(floatStream)
				.forEach(e -> dataMatrix.asEditableMatrix().putColumn((int) e.getIndex(),
						matrixFactory.createMatrixFromRowsByRowsArray(featureCount, 1, e.getValue())));

		return new NeuronsActivationImpl(dataMatrix, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}
	
	
}
