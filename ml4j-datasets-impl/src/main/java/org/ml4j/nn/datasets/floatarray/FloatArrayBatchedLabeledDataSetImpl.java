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

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.BatchedLabeledDataSetImpl;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationLabeledDataSet;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationLabeledDataSetImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.streams.Streamable;

public class FloatArrayBatchedLabeledDataSetImpl extends BatchedLabeledDataSetImpl<float[], float[]>
		implements FloatArrayBatchedLabeledDataSet {

	private int featureCount;
	private int labelFeatureCount;

	public FloatArrayBatchedLabeledDataSetImpl(Supplier<Stream<DataBatch<LabeledData<float[], float[]>>>> dataSupplier,
			int featureCount, int labelFeatureCount) {
		super(dataSupplier);
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
	public NeuronsActivationLabeledDataSet toNeuronsActivationLabeledDataSet(MatrixFactory matrixFactory,
			NeuronsActivationFormat<?> format) {
		return new NeuronsActivationLabeledDataSetImpl(
				() -> stream().map(batch -> createLabeledData(matrixFactory, batch, format)));
	}

	private LabeledData<NeuronsActivation, NeuronsActivation> createLabeledData(MatrixFactory matrixFactory,
			DataBatch<LabeledData<float[], float[]>> batch, NeuronsActivationFormat<?> format) {
		return createFloatArrayLabeledDataBatch(batch).toNeuronsActivations(matrixFactory, format);
	}

	private FloatArrayLabeledDataBatch createFloatArrayLabeledDataBatch(
			DataBatch<LabeledData<float[], float[]>> batch) {

		LabeledData<DataBatch<float[]>, DataBatch<float[]>> dataBatches = convertBatchOfLabeledDataToLabeledDataOfBatches(
				batch);

		return new FloatArrayLabeledDataBatchImpl(dataBatches.getData(), dataBatches.getLabel(), featureCount,
				labelFeatureCount);
	}

	private <E, L> LabeledData<DataBatch<E>, DataBatch<L>> convertBatchOfLabeledDataToLabeledDataOfBatches(
			DataBatch<LabeledData<E, L>> dataBatchOfLabeledData) {

		Streamable<E> data = dataBatchOfLabeledData.map(batch -> batch.getData());
		Streamable<L> labels = dataBatchOfLabeledData.map(batch -> batch.getLabel());

		DataBatch<E> dataBatch = new DataBatchImpl<>(data.stream(), dataBatchOfLabeledData.size());
		DataBatch<L> labelsBatch = new DataBatchImpl<>(labels.stream(), dataBatchOfLabeledData.size());

		return new LabeledDataImpl<>(dataBatch, labelsBatch);
	}

}
