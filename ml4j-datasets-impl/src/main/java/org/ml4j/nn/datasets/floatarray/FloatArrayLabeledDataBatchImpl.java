package org.ml4j.nn.datasets.floatarray;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataBatchImpl;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

import com.codepoetics.protonpack.StreamUtils;

public class FloatArrayLabeledDataBatchImpl extends LabeledDataBatchImpl<float[], float[]>
		implements FloatArrayLabeledDataBatch {

	private int featureCount;
	private int labelFeatureCount;
	private int batchSize;

	public FloatArrayLabeledDataBatchImpl(int featureCount, int labelFeatureCount, int batchSize) {
		super(batchSize);
		this.featureCount = featureCount;
		this.batchSize = batchSize;
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
		this.labelFeatureCount = labelFeatureCount;
	}
	
	

	public FloatArrayLabeledDataBatchImpl(DataBatch<float[]> dataBatch, DataBatch<float[]> labelBatch, int featureCount, int labelFeatureCount) {
		super(dataBatch, labelBatch);
		this.featureCount = featureCount;
		this.batchSize = dataBatch.size();
		if (dataBatch.size() != labelBatch.size()) {
			throw new IllegalArgumentException();

		}
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		this.labelFeatureCount = labelFeatureCount;
	}

	public LabeledData<NeuronsActivation, NeuronsActivation> getNeuronActivations(MatrixFactory matrixFactory) {

		Matrix data = matrixFactory.createMatrix(batchSize, featureCount);
		Matrix labels = matrixFactory.createMatrix(batchSize, labelFeatureCount);

		StreamUtils.zipWithIndex(stream()).forEach(d -> addToMatrices(matrixFactory, data, labels,
				d.getValue().getData(), d.getValue().getLabel(), (int) d.getIndex()));

		return new LabeledDataImpl<>(
				new NeuronsActivationImpl(data.transpose(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
				new NeuronsActivationImpl(labels.transpose(),
						NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET));
	}

	private void addToMatrices(MatrixFactory matrixFactory, Matrix dataMatrix, Matrix labelsMatrix, float[] data,
			float[] labels, int row) {
		addToMatrix(matrixFactory, dataMatrix, data, featureCount, row);
		addToMatrix(matrixFactory, labelsMatrix, labels, labelFeatureCount, row);
	}

	private void addToMatrix(MatrixFactory matrixFactory, Matrix matrix, float[] data, int featureCount, int row) {
		matrix.asEditableMatrix().putRow(row, matrixFactory.createMatrixFromRowsByRowsArray(1, featureCount, data));
	}

	@Override
	public FloatArrayDataBatch getDataSet() {
		FloatArrayDataBatch dataBatch = new FloatArrayDataBatchImpl(featureCount, batchSize);
		super.getDataSet().stream().forEach(f -> dataBatch.add(f));
		return dataBatch;
	}

	@Override
	public FloatArrayDataBatch getLabelsSet() {
		FloatArrayDataBatch dataBatch = new FloatArrayDataBatchImpl(labelFeatureCount, batchSize);
		super.getLabelsSet().stream().forEach(f -> dataBatch.add(f));
		return dataBatch;
	}
	
	
	@Override
	public LabeledData<NeuronsActivation, NeuronsActivation> toNeuronsActivations(MatrixFactory matrixFactory) {
		return new LabeledDataImpl<>(getDataSet().toNeuronsActivation(matrixFactory),
				getLabelsSet().toNeuronsActivation(matrixFactory));
	}

}
