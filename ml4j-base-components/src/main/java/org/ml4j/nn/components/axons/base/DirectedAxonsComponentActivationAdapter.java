package org.ml4j.nn.components.axons.base;

import java.util.Date;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentActivationAdapter implements DirectedAxonsComponentActivation {

	private DirectedAxonsComponentActivation delegated;
	private String name;
	
	public DirectedAxonsComponentActivationAdapter(DirectedAxonsComponentActivation delegated, String name) {
		this.delegated = delegated;
		this.name = name;
	}
	
	@Override
	public List<? extends DefaultChainableDirectedComponentActivation> decompose() {
		return delegated.decompose();
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		long startTime = new Date().getTime();
		DirectedComponentGradient<NeuronsActivation> grad =  delegated.backPropagate(gradient);
		long endTime = new Date().getTime();
		long timeTaken = endTime - startTime;
		DefaultChainableDirectedComponentAdapter.addTime(timeTaken, "bp:" + name);
		return grad;
	}

	@Override
	public NeuronsActivation getOutput() {
		return delegated.getOutput();
	}

	@Override
	public double getAverageRegularisationCost() {
		return delegated.getAverageRegularisationCost();
	}

	@Override
	public DirectedAxonsComponent<?, ?, ?> getAxonsComponent() {
		return delegated.getAxonsComponent();
	}

	@Override
	public float getTotalRegularisationCost() {
		return delegated.getTotalRegularisationCost();
	}

}
