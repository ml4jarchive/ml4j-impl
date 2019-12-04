package org.ml4j.nn.components.axons;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainActivationImpl;
import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentChainImpl<L extends Neurons, R extends Neurons> extends DirectedComponentChainBaseImpl<NeuronsActivation, DirectedAxonsComponent<L, R>, DirectedAxonsComponentActivation, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>> implements DirectedAxonsComponentChain<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedAxonsComponentChainImpl(
			List<DirectedAxonsComponent<L, R>> components) {
		super(components);
	}

	@Override
	protected DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation> createChainActivation(
			List<DirectedAxonsComponentActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DirectedComponentChainActivationImpl<>(componentActivations, inFlightInput);
	}

	//@Override
	//public DirectedSynapsesChain<L, R> dup() {
	//	return new DirectedSynapsesChainImpl<>(components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	//}

	//@Override
	//protected ChainableDirectedComponentActivation<NeuronsActivation> forwardPropagate(NeuronsActivation input,
	//		DirectedAxonsComponent<L, R> component, int componentIndex,
	//		DirectedComponentsContext context) {
	//	return component.forwardPropagate(input, context.getContext(component));
	//}

	//@Override
	//protected DirectedSynapsesChain<L, R> createDirectedComponentChain(List<DirectedSynapses<L, R>> components) {
///		return new DirectedSynapsesChainImpl<>(components);
	//}

	
}
