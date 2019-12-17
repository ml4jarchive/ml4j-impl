package org.ml4j.nn.datasets;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.floatarray.FloatArrayBatchedDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatch;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatchImpl;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationDataSet;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationDataSetImpl;

public class FloatArrayBatchedDataSetImpl extends BatchedDataSetImpl<float[]> implements FloatArrayBatchedDataSet {

	private int featureCount;
	
	public FloatArrayBatchedDataSetImpl(Supplier<Stream<DataBatch<float[]>>> dataSupplier, int featureCount) {
		super(dataSupplier);
		this.featureCount = featureCount;
	}

	@Override
	public NeuronsActivationDataSet toNeuronsActivationDataSet(MatrixFactory matrixFactory) {		
		return new NeuronsActivationDataSetImpl(() -> stream().map(batch -> createFloatArrayDataBatch(batch)).map(batch -> batch.toNeuronsActivation(matrixFactory)));
	}
	
	private FloatArrayDataBatch createFloatArrayDataBatch(DataBatch<float[]> batch)  {
		return new FloatArrayDataBatchImpl(batch.stream(), featureCount, batch.size());
	}
}
