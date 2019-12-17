package org.ml4j.nn.datasets.floatarray;

import java.io.File;
import java.io.ObjectInputStream;
import java.util.function.Supplier;

import org.ml4j.nn.datasets.FloatArrayBatchedDataSetImpl;
import org.ml4j.nn.datasets.util.ObjectInputStreamDataSet;

public class FloatArrayInputStreamDataSet extends ObjectInputStreamDataSet<float[]> implements FloatArrayDataSet {

	private int featureCount;
	
	public FloatArrayInputStreamDataSet(File file, int featureCount) {
		super(file, float[].class);
		this.featureCount = featureCount;
	}

	public FloatArrayInputStreamDataSet(Supplier<ObjectInputStream> objectInputStreamSupplier, int featureCount) {
		super(objectInputStreamSupplier, float[].class);
		this.featureCount = featureCount;
	}

	@Override
	public FloatArrayBatchedDataSet toBatchedDataSet(int batchSize) {
		return new FloatArrayBatchedDataSetImpl(super.toBatchedDataSet(batchSize), featureCount);
	}

	
}
