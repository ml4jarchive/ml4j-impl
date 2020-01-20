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

import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.floatarray.FloatArrayBatchedDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatch;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatchImpl;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationDataSet;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationDataSetImpl;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

public class FloatArrayBatchedDataSetImpl extends BatchedDataSetImpl<float[]> implements FloatArrayBatchedDataSet {

	private int featureCount;

	public FloatArrayBatchedDataSetImpl(Supplier<Stream<DataBatch<float[]>>> dataSupplier, int featureCount) {
		super(dataSupplier);
		this.featureCount = featureCount;
	}

	@Override
	public NeuronsActivationDataSet toNeuronsActivationDataSet(MatrixFactory matrixFactory,
			NeuronsActivationFeatureOrientation featureOrientation) {
		return new NeuronsActivationDataSetImpl(() -> stream().map(batch -> createFloatArrayDataBatch(batch))
				.map(batch -> batch.toNeuronsActivation(matrixFactory, featureOrientation)));
	}

	private FloatArrayDataBatch createFloatArrayDataBatch(DataBatch<float[]> batch) {
		return new FloatArrayDataBatchImpl(batch.stream(), featureCount, batch.size());
	}
}
