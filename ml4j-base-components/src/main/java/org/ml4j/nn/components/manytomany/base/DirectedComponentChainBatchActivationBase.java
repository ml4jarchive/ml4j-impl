package org.ml4j.nn.components.manytomany.base;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class DirectedComponentChainBatchActivationBase implements DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation>{

	protected List<DefaultDirectedComponentChainActivation> activations;
	
	public DirectedComponentChainBatchActivationBase(List<DefaultDirectedComponentChainActivation> activations) {
		this.activations = activations;
	}

	@Override
	public List<NeuronsActivation> getOutput() {
		return activations.stream().map(a -> a.getOutput()).collect(Collectors.toList());
	}

	@Override
	public List<DefaultDirectedComponentChainActivation> getActivations() {
		return activations;
	}

}
