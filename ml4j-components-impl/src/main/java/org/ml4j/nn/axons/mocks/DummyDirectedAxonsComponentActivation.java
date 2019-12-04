package org.ml4j.nn.axons.mocks;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedAxonsComponentActivation implements DirectedAxonsComponentActivation {

	private DirectedAxonsComponent<?, ?> directedAxonsComponent;
	private NeuronsActivation input;
	private NeuronsActivation output;
	
	public DummyDirectedAxonsComponentActivation(DirectedAxonsComponent<?, ?> directedAxonsComponent, NeuronsActivation input, 
			NeuronsActivation output) {
		this.directedAxonsComponent = directedAxonsComponent;
		this.input = input;
		this.output = output;
	}

	@Override
	public List<ChainableDirectedComponentActivation<NeuronsActivation>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		return new DummyDirectedComponentGradient(input);
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public DirectedAxonsComponent<?, ?> getAxonsComponent() {
		return directedAxonsComponent;
	}

	@Override
	public float getTotalRegularisationCost() {
		return 0;
	}

	@Override
	public double getAverageRegularisationCost() {
		return 0;
	}

}
