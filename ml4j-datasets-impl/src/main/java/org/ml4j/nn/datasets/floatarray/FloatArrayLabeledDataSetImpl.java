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
package org.ml4j.nn.datasets.floatarray;

import java.util.Optional;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.DataSet;
import org.ml4j.nn.datasets.DataSetImpl;
import org.ml4j.nn.datasets.FeatureExtractionErrorMode;
import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.util.StreamUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FloatArrayLabeledDataSetImpl implements FloatArrayLabeledDataSet {

	private static final Logger LOGGER = LoggerFactory.getLogger(FloatArrayLabeledDataSetImpl.class);

	private int featureCount;
	private int labelFeatureCount;
	private Supplier<Stream<LabeledData<float[], float[]>>> streamSupplier;
	
	public FloatArrayLabeledDataSetImpl(Supplier<Stream<LabeledData<float[], float[]>>> streamSupplier, int featureCount, int labelFeatureCount) {
		this.streamSupplier = streamSupplier;
		this.featureCount = featureCount;
		this.labelFeatureCount = labelFeatureCount;
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
		if (labelFeatureCount == 0) {
			throw new IllegalArgumentException();
		}
	}
	
	@Override
	public DataSet<float[]> getDataSet() {
		return new DataSetImpl<>(() -> stream().map(l -> l.getData()));
	}

	@Override
	public Stream<float[]> getLabels() {
		return stream().map(l -> l.getLabel());
	}

	@Override
	public DataSet<float[]> getLabelsSet() {
		return new DataSetImpl<>(() -> stream().map(l -> l.getLabel()));
	}

	@Override
	public Stream<LabeledData<float[], float[]>> stream() {
		return streamSupplier.get();
	}

	@Override
	public FloatArrayLabeledDataSet toFloatArrayLabeledDataSet(FeatureExtractor<float[]> featureExtractor,
			FeatureExtractor<float[]> labelMapper, FeatureExtractionErrorMode featureExtractionErrorMode) {
		return new FloatArrayLabeledDataSetImpl(() -> stream().map(l -> toLabeledFloatArray(featureExtractor, labelMapper, l, featureExtractionErrorMode)).filter(Optional::isPresent).map(Optional::get), featureExtractor.getFeatureCount(), labelMapper.getFeatureCount());
	}
	
	private <E, L> Optional<LabeledData<float[], float[]>> toLabeledFloatArray(FeatureExtractor<E> featureExtractor, FeatureExtractor<L> labelMapper, 
			LabeledData<E, L> labeledData, 
			FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			return Optional.of(new LabeledDataImpl<>(featureExtractor.getFeatures(labeledData.getData()), labelMapper.getFeatures(labeledData.getLabel())));
		} catch (FeatureExtractionException e) {
			if (featureExtractionErrorMode == FeatureExtractionErrorMode.LOG_WARNING) {
				LOGGER.warn("Ignoring data element due to feature extraction failure", e);
			} else if (featureExtractionErrorMode == FeatureExtractionErrorMode.RAISE_EXCEPTION) {
				throw new FeatureExtractionRuntimeException("Unable to convert labeled data to labeled float arrays", e);
			}
			return Optional.empty();
		}
	}

	

	@Override
	public FloatArrayBatchedLabeledDataSet toBatchedLabeledDataSet(int batchSize) {			
		return new FloatArrayBatchedLabeledDataSetImpl(() -> StreamUtil.partition(stream().map(l -> new LabeledDataImpl<>(l.getData(), l.getLabel())), batchSize), featureCount, labelFeatureCount);
	}

}
