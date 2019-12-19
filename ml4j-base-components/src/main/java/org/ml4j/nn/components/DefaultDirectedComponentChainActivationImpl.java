package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainActivationImpl extends DirectedComponentChainActivationImpl<NeuronsActivation, DefaultChainableDirectedComponentActivation>
		implements DefaultDirectedComponentChainActivation {

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		List<ChainableDirectedComponentActivation<?>> acts = new ArrayList<>();
		acts.addAll(activations);
		return activations;
	}

	public DefaultDirectedComponentChainActivationImpl(List<DefaultChainableDirectedComponentActivation> activations,
			NeuronsActivation output) {
		super(activations, output);
	}


}
