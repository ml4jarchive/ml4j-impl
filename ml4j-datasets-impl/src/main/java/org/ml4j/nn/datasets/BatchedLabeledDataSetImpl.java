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

import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.floatarray.FloatArrayBatchedLabeledDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayBatchedLabeledDataSetImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchedLabeledDataSetImpl<E, L> extends BatchedDataSetImpl<LabeledData<E, L>>
		implements BatchedLabeledDataSet<E, L> {

	private static final Logger LOGGER = LoggerFactory.getLogger(BatchedLabeledDataSetImpl.class);
	
	public BatchedLabeledDataSetImpl(Supplier<Stream<DataBatch<LabeledData<E, L>>>> dataSupplier) {
		super(dataSupplier);
	}

	@Override
	public FloatArrayBatchedLabeledDataSet toFloatArrayBatchedLabeledDataSet(FeatureExtractor<E> featureExtractor,
			FeatureExtractor<L> labelMapper, FeatureExtractionErrorMode featureExtractionErrorMode) {
		
		return new FloatArrayBatchedLabeledDataSetImpl(() -> stream().map(batch -> extract(batch, featureExtractor, labelMapper, featureExtractionErrorMode)), featureExtractor.getFeatureCount(), labelMapper.getFeatureCount());
	}


	private DataBatch<LabeledData<float[], float[]>> extract(DataBatch<LabeledData<E, L>> batch, FeatureExtractor<E> featureExtractor,
			FeatureExtractor<L> labelMapper, FeatureExtractionErrorMode featureExtractionErrorMode) {
		Stream<LabeledData<float[], float[]>> s = batch.stream().map(l -> 
		new LabeledDataImpl<Optional<float[]>, Optional<float[]>>(getFeatures(l.getData(), featureExtractor, featureExtractionErrorMode), getFeatures(l.getLabel(), labelMapper, featureExtractionErrorMode))).
				filter(o -> o.getData().isPresent() && o.getLabel().isPresent()).map(o -> new LabeledDataImpl<>(o.getData().get(), o.getLabel().get()));
		return new DataBatchImpl<LabeledData<float[], float[]>>(s, batch.size());
	}
	
	private <T> Optional<float[]> getFeatures(T element, FeatureExtractor<T> featureExtractor, FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			return Optional.of(featureExtractor.getFeatures(element));
		} catch (FeatureExtractionException e) {
			if (featureExtractionErrorMode == FeatureExtractionErrorMode.LOG_WARNING) {
				LOGGER.warn("Ignoring data element due to feature extraction failure", e);
			} else if (featureExtractionErrorMode  == FeatureExtractionErrorMode.RAISE_EXCEPTION) {
				throw new FeatureExtractionRuntimeException("Unable to obtain features", e);
			}
			return Optional.empty();
		}
	}
}
