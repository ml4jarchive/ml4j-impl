package org.ml4j.nn.datasets;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.floatarray.FloatArrayBatchedDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatch;

public class BatchedDataSetImpl<E> extends DataSetImpl<DataBatch<E>> implements BatchedDataSet<E> {

	public BatchedDataSetImpl(Supplier<Stream<DataBatch<E>>> dataSupplier) {
		super(dataSupplier);
	}

	@Override
	public FloatArrayBatchedDataSet toFloatArrayBatchedDataSet(FeatureExtractor<E> featureExtractor, FeatureExtractionErrorMode featureExtractionErrorMode) {
		return new FloatArrayBatchedDataSetImpl(transform(batchStream -> batchStream.map(batch -> createFloatArrayDataBatch(batch, featureExtractor, featureExtractionErrorMode))), featureExtractor.getFeatureCount());
	}
	
	private FloatArrayDataBatch createFloatArrayDataBatch(DataBatch<E> batch, FeatureExtractor<E> featureExtractor, FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			FloatArrayDataBatch dataBatch =  batch.toFloatArrayDataBatch(featureExtractor, featureExtractionErrorMode);	
			return dataBatch;
		} catch (FeatureExtractionException e) {
			throw new FeatureExtractionRuntimeException("Unable to convert batch to float array batch", e);
		}
	}

}
