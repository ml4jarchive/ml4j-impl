package org.ml4j.nn.synapses;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphActivationImpl;
import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.GenericOneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesBipoleGraphActivationImpl extends DirectedComponentsBipoleGraphActivationImpl<NeuronsActivation, DirectedSynapsesChainActivation>
	implements DirectedSynapsesBipoleGraphActivation {

	public DirectedSynapsesBipoleGraphActivationImpl(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> inputLinkActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation> edgesActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> outputLinkActivation) {
		super(inputLinkActivation, edgesActivation, outputLinkActivation);
	}

}
