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
