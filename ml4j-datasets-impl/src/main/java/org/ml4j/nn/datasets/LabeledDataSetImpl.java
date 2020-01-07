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
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSetImpl;
import org.ml4j.nn.datasets.util.StreamUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codepoetics.protonpack.StreamUtils;


public class LabeledDataSetImpl<E, L> implements LabeledDataSet<E, L> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DataBatchImpl.class);

	
	private Supplier<Stream<LabeledData<E, L>>> labeledDataSupplier;

	public LabeledDataSetImpl(Supplier<Stream<LabeledData<E, L>>> labeledDataSupplier) {
		this.labeledDataSupplier = labeledDataSupplier;
	}

	public <O> LabeledDataSetImpl(DataSet<LabeledData<E, O>> labeledDataSet, DataLabeler<O, L> dataLabeler) {
		this.labeledDataSupplier = () -> StreamUtils.zipWithIndex(labeledDataSet.stream())
				.map(i -> new LabeledDataImpl<>(i.getValue().getData(),
						dataLabeler.getLabel(i.getIndex(), i.getValue().getLabel())));

	}

	@Override
	public DataSet<E> getDataSet() {
		return new DataSetImpl<>(() -> labeledDataSupplier.get().map(l -> l.getData()));
	}

	@Override
	public Stream<L> getLabels() {
		return labeledDataSupplier.get().map(l -> l.getLabel());
	}

	@Override
	public Stream<LabeledData<E, L>> stream() {
		return labeledDataSupplier.get();
	}

	@Override
	public DataSet<L> getLabelsSet() {
		return new DataSetImpl<>(() -> labeledDataSupplier.get().map(l -> l.getLabel()));
	}

	@Override
	public BatchedLabeledDataSet<E, L> toBatchedLabeledDataSet(int batchSize) {

		Stream<DataBatch<LabeledData<E, L>>> dataBatchStream = StreamUtil.partition(stream(), batchSize);

		return new BatchedLabeledDataSetImpl<E, L>(() -> dataBatchStream
				.map(batchOfLabeledData -> batchOfLabeledData));
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
				throw new FeatureExtractionRuntimeException("Unable to convert labeled data to labeled float arrays", e);
			}
			return Optional.empty();
		}
	}
}
