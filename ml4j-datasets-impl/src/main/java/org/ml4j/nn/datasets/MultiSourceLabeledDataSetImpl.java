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
import java.util.stream.Stream;

import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSetImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codepoetics.protonpack.StreamUtils;

public class MultiSourceLabeledDataSetImpl<E, L> implements LabeledDataSet<E, L> {

	private static final Logger LOGGER = LoggerFactory.getLogger(MultiSourceLabeledDataSetImpl.class);
	
	private DataSet<E> dataSet;
	private DataSet<L> labelSet;

	public MultiSourceLabeledDataSetImpl(DataSet<E> dataSet, DataSet<L> labelSet) {
		this.dataSet = dataSet;
		this.labelSet = labelSet;
	}

	@Override
	public DataSet<E> getDataSet() {
		return dataSet;
	}

	@Override
	public DataSet<L> getLabelsSet() {
		return labelSet;
	}

	@Override
	public Stream<L> getLabels() {
		return labelSet.stream();
	}

	@Override
	public Stream<LabeledData<E, L>> stream() {
		return StreamUtils.zip(dataSet.stream(), labelSet.stream(), (l, r) -> createLabeledData(l, r))
				.filter(Optional::isPresent).map(Optional::get);
	}

	private Optional<LabeledData<E, L>> createLabeledData(E element, L label) {
		if (element == null || label == null) {
			return Optional.empty();
		} else {
			return Optional.of(new LabeledDataImpl<>(element, label));
		}
	}

	@Override
	public BatchedLabeledDataSet<E, L> toBatchedLabeledDataSet(int batchSize) {

		BatchedDataSet<E> batchedDataSet = dataSet.toBatchedDataSet(batchSize);
		BatchedDataSet<L> batchedLabelSet = labelSet.toBatchedDataSet(batchSize);

		StreamUtils.zip(batchedDataSet.stream(), batchedLabelSet.stream(), (l,
				r) -> new LabeledDataImpl<>(new DataBatchImpl<>(l.get(), batchSize), new DataBatchImpl<>(r.get(), 1)));

		Stream<DataBatch<LabeledData<E, L>>> s = StreamUtils.zip(batchedDataSet.stream(), batchedLabelSet.stream(),
				(l, r) -> toLabeledData(l, r));

		return new BatchedLabeledDataSetImpl<>(() -> s);
	}

	private DataBatch<LabeledData<E, L>> toLabeledData(DataBatch<E> data, DataBatch<L> labels) {

		if (data.size() != labels.size()) {
			throw new IllegalArgumentException();
		}
		
		return new DataBatchImpl<>(
				StreamUtils.zip(data.stream(), labels.stream(), (l, r) -> new LabeledDataImpl<>(l, r)), data.size());

	}


	@Override
	public FloatArrayLabeledDataSet toFloatArrayLabeledDataSet(FeatureExtractor<E> featureExtractor, FeatureExtractor<L> labelMapper,
			FeatureExtractionErrorMode featureExtractionErrorMode) {
		return new FloatArrayLabeledDataSetImpl(() -> stream().map(l -> toLabeledFloatArray(featureExtractor, labelMapper, l, 
				featureExtractionErrorMode))
				.filter(Optional::isPresent).map(Optional::get), featureExtractor.getFeatureCount(), labelMapper.getFeatureCount());
	}
	
	private Optional<LabeledData<float[], float[]>> toLabeledFloatArray(FeatureExtractor<E> featureExtractor, FeatureExtractor<L> labelMapper, 
			LabeledData<E, L> labeledData, 
			FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			return Optional.of(new LabeledDataImpl<>(featureExtractor.getFeatures(labeledData.getData()), labelMapper.getFeatures(labeledData.getLabel())));
		} catch (FeatureExtractionException e) {
			if (featureExtractionErrorMode == FeatureExtractionErrorMode.LOG_WARNING) {
				LOGGER.warn("Ignoring data element due to feature extraction failure", e);
			} else if (featureExtractionErrorMode == FeatureExtractionErrorMode.RAISE_EXCEPTION) {
				throw new FeatureExtractionRuntimeException("Unable to convert labeled data to labeled float array", e);
			}
			return Optional.empty();

		}
	}

	
	
}
