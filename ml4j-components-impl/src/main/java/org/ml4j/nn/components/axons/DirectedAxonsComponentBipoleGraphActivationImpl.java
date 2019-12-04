package org.ml4j.nn.components.axons;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphActivationImpl;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentBipoleGraphActivationImpl extends DirectedComponentsBipoleGraphActivationImpl<NeuronsActivation, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>>
	implements DirectedAxonsComponentBipoleGraphActivation {

	public DirectedAxonsComponentBipoleGraphActivationImpl(
			OneToManyDirectedComponentActivation<NeuronsActivation> inputLinkActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>> edgesActivation,
			ManyToOneDirectedComponentActivation<NeuronsActivation> outputLinkActivation) {
		super(inputLinkActivation, edgesActivation, outputLinkActivation);
	}

}
