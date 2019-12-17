package org.ml4j.nn.components;

import java.util.List;

public abstract class GenericOneToManyDirectedComponentActivationImpl<I> implements GenericOneToManyDirectedComponentActivation<I> {

	protected List<I> activations;
	
	public GenericOneToManyDirectedComponentActivationImpl(List<I> activations) { 
		this.activations = activations;
	}
	
	@Override
	public DirectedComponentGradient<I> backPropagate(DirectedComponentGradient<List<I>> outerGradient) {
		return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(), getBackPropagatedGradient(outerGradient.getOutput()));
	}
	
	protected abstract I getBackPropagatedGradient(List<I> gradient);

	@Override
	public List<I> getOutput() {
		return activations;
	}
}
