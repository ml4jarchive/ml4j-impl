package org.ml4j.nn.components;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentsBipoleGraphActivationImpl
		extends DirectedComponentsBipoleGraphActivationImpl<NeuronsActivation, DefaultDirectedComponentChainActivation> implements DefaultDirectedComponentBipoleGraphActivation {

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return edgesActivation.getActivations().stream().flatMap(a -> a.decompose().stream())
				.collect(Collectors.toList());
	}

	public DefaultDirectedComponentsBipoleGraphActivationImpl(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> inputLinkActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation> edgesActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> outputLinkActivation) {
		super(inputLinkActivation, edgesActivation, outputLinkActivation);
	}
	
	

}
