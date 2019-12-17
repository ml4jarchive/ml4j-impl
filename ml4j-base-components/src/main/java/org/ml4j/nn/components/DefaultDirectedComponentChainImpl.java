package org.ml4j.nn.components;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainImpl extends DirectedComponentChainBaseImpl<NeuronsActivation, DefaultChainableDirectedComponent<?, ?>, DefaultChainableDirectedComponentActivation, DefaultDirectedComponentChainActivation> implements DefaultDirectedComponentChain{

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainImpl(List<? extends DefaultChainableDirectedComponent<?, ?>> components) {
		super(components);
	}

	@Override
	public DefaultDirectedComponentChain dup() {
		return new DefaultDirectedComponentChainImpl(components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

	@Override
	protected DefaultDirectedComponentChainActivation createChainActivation(
			List<DefaultChainableDirectedComponentActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DefaultDirectedComponentChainActivationImpl(componentActivations, inFlightInput);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		// TODO Auto-generated method stub
		// 		return components.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());

		return null;
	}

	
	
}
