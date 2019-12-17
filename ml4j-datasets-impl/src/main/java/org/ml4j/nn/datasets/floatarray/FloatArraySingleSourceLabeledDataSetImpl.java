package org.ml4j.nn.datasets.floatarray;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.DataLabeler;
import org.ml4j.nn.datasets.DataSet;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataSetImpl;

public class FloatArraySingleSourceLabeledDataSetImpl extends LabeledDataSetImpl<float[], float[]> implements FloatArrayLabeledDataSet {

	private int featureCount;
	private int labelFeatureCount;
	
	public <O> FloatArraySingleSourceLabeledDataSetImpl(DataSet<LabeledData<float[], O>> labeledDataSet,
			DataLabeler<O, float[]> dataLabeler, int featureCount, int labelFeatureCount) {
		super(labeledDataSet, dataLabeler);
		this.featureCount = featureCount;
		this.labelFeatureCount = labelFeatureCount;
	}

	public FloatArraySingleSourceLabeledDataSetImpl(Supplier<Stream<LabeledData<float[], float[]>>> labeledDataSupplier, int featureCount, int labelFeatureCount) {
		super(labeledDataSupplier);
		this.featureCount = featureCount;
		this.labelFeatureCount = labelFeatureCount;
	}

	@Override
	public FloatArrayBatchedLabeledDataSet toBatchedLabeledDataSet(int batchSize) {
		return new FloatArrayBatchedLabeledDataSetImpl(super.toBatchedLabeledDataSet(batchSize), featureCount, labelFeatureCount);
	}

	
}
