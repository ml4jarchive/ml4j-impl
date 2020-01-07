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
package org.ml4j.nn.datasets.neuronsactivation;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.DataLabeler;
import org.ml4j.nn.datasets.DataSet;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataSetImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class NeuronsActivationLabeledDataSetImpl extends LabeledDataSetImpl<NeuronsActivation, NeuronsActivation>
		implements NeuronsActivationLabeledDataSet {

	public <O> NeuronsActivationLabeledDataSetImpl(DataSet<LabeledData<NeuronsActivation, O>> labeledDataSet,
			DataLabeler<O, NeuronsActivation> dataLabeler) {
		super(labeledDataSet, dataLabeler);
	}

	public NeuronsActivationLabeledDataSetImpl(
			Supplier<Stream<LabeledData<NeuronsActivation, NeuronsActivation>>> labeledDataSupplier) {
		super(labeledDataSupplier);
	}

}
