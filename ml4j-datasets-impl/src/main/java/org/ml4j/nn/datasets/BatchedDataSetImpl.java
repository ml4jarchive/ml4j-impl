/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
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
