package org.ml4j.nn.axons.mocks;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedComponentGradient implements DirectedComponentGradient<NeuronsActivation> {
	
	private NeuronsActivation output;
	
	public DummyDirectedComponentGradient(NeuronsActivation output) {
		this.output = output;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public void addTotalTrainableAxonsGradient(Supplier<AxonsGradient> axonsGradient) {
		// No-op
	}

	@Override
	public List<Supplier<AxonsGradient>> getTotalTrainableAxonsGradients() {
		return Arrays.asList(() -> null);
	}

	@Override
	public List<Supplier<AxonsGradient>> getAverageTrainableAxonsGradients() {
		return Arrays.asList(() -> null);
	}

}
