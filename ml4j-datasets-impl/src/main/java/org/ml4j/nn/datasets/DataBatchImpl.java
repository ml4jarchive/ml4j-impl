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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatch;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatchImpl;
import org.ml4j.nn.datasets.util.StreamUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataBatchImpl<E> implements DataBatch<E> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DataBatchImpl.class);

	private List<E> dataList;

	public DataBatchImpl(int batchSize) {
		this.dataList = new ArrayList<>(batchSize);
	}

	public DataBatchImpl(Stream<E> data, int batchSize) {
		this.dataList = new ArrayList<>(batchSize);
		data.forEach(e -> dataList.add(e));
		if (batchSize != size()) {
			throw new IllegalStateException();
		}
	}

	@Override
	public Stream<E> stream() {
		return dataList.stream();
	}

	@Override
	public void add(E data) {
		dataList.add(data);
	}

	@Override
	public int size() {
		return dataList.size();
	}

	@Override
	public boolean isEmpty() {
		return dataList.isEmpty();
	}

	@Override
	public BatchedDataSet<E> toBatchedDataSet(int batchSize) {
		return new BatchedDataSetImpl<E>(() -> StreamUtil.partition(stream(), batchSize));
	}

	private Optional<float[]> getFeatures(E element, FeatureExtractor<E> featureExtractor,
			FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			return Optional.of(featureExtractor.getFeatures(element));
		} catch (FeatureExtractionException e) {
			if (featureExtractionErrorMode == FeatureExtractionErrorMode.LOG_WARNING) {
				LOGGER.warn("Ignoring data element due to feature extraction failure", e);
			} else if (featureExtractionErrorMode == FeatureExtractionErrorMode.RAISE_EXCEPTION) {
				throw new FeatureExtractionRuntimeException("Unable to obtain features from element", e);
			}
			return Optional.empty();
		}
	}

	@Override
	public FloatArrayDataBatch toFloatArrayDataBatch(FeatureExtractor<E> featureExtractor,
			FeatureExtractionErrorMode featureExtractionErrorMode) throws FeatureExtractionException {
		return new FloatArrayDataBatchImpl(
				stream().map(e -> getFeatures(e, featureExtractor, featureExtractionErrorMode))
						.filter(Optional::isPresent).map(e -> e.get()),
				featureExtractor.getFeatureCount(), size());
	}

}
