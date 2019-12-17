package org.ml4j.nn.datasets.neuronsactivation;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.DataSetImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class NeuronsActivationDataSetImpl extends DataSetImpl<NeuronsActivation> implements NeuronsActivationDataSet {

	public NeuronsActivationDataSetImpl(Supplier<Stream<NeuronsActivation>> dataSupplier) {
		super(dataSupplier);
	}

}
