package org.ml4j.nn.layers;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedLayerChainImpl<L extends DirectedLayer<?, ?>> extends DirectedComponentChainBaseImpl<NeuronsActivation, L, DirectedLayerActivation, DirectedLayerChainActivation> implements DirectedLayerChain<L> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedLayerChainImpl(
			List<L> components) {
		super(components);
	}

	@Override
	protected DirectedLayerChainActivation createChainActivation(
			List<DirectedLayerActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DirectedLayerChainActivationImpl(componentActivations);
	}

	@SuppressWarnings("unchecked")
	@Override
	public DirectedComponentChain<NeuronsActivation, L, DirectedLayerActivation, DirectedLayerChainActivation> dup() {
		return new DirectedLayerChainImpl<L>((List<L>) this.components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}	
}
