package org.ml4j.nn.datasets.floatarray;

import org.ml4j.nn.datasets.BatchedDataSet;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.datasets.DataSet;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.MultiSourceLabeledDataSetImpl;

import com.codepoetics.protonpack.StreamUtils;

public class FloatArrayMultiSourceLabeledDataSetImpl extends MultiSourceLabeledDataSetImpl<float[], float[]> implements FloatArrayLabeledDataSet {

	private int featureCount;
	private int labelFeatureCount;
	
	public FloatArrayMultiSourceLabeledDataSetImpl(DataSet<float[]> dataSet, DataSet<float[]> labelSet, int featureCount, int labelFeatureCount) {
		super(dataSet, labelSet);
		this.featureCount = featureCount;
		this.labelFeatureCount = labelFeatureCount;
	}
	
	@Override
	public FloatArrayBatchedLabeledDataSet toBatchedLabeledDataSet(int batchSize) {
					
		BatchedDataSet<float[]> batchedDataSet = getDataSet().toBatchedDataSet(batchSize);
		BatchedDataSet<float[]> batchedLabelSet = getLabelsSet().toBatchedDataSet(batchSize);

		return new FloatArrayBatchedLabeledDataSetImpl(() -> StreamUtils.zip(batchedDataSet.stream(), batchedLabelSet.stream(), (l,r) -> createLabeledDataBatch(l, r)), featureCount, labelFeatureCount);

	}
	
	private DataBatch<LabeledData<float[], float[]>> createLabeledDataBatch(DataBatch<float[]> data, DataBatch<float[]> labels) {
		if (data.size() != labels.size()) {
			throw new IllegalArgumentException();
		}
		return new DataBatchImpl<>(StreamUtils.zip(data.stream(), labels.stream(), (l,r) -> new LabeledDataImpl<>(l, r)), data.size());
	}

	
}
