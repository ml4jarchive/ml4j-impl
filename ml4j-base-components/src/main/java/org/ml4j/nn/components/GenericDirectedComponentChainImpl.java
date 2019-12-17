package org.ml4j.nn.components;

import java.util.List;

public abstract class GenericDirectedComponentChainImpl<I, A extends ChainableDirectedComponentActivation<I>> extends DirectedComponentChainBaseImpl<I, ChainableDirectedComponent<I, ? extends A, ?> ,A, DirectedComponentChainActivation<I, A>> implements GenericDirectedComponentChain<I, A> {

	@Override
	protected DirectedComponentChainActivation<I, A> createChainActivation(List<A> componentActivations,
			I inFlightInput) {
		return new DirectedComponentChainActivationImpl<>(componentActivations, inFlightInput);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public GenericDirectedComponentChainImpl(List<? extends ChainableDirectedComponent<I, ? extends A, ?>> components) {
		super(components);
	}
}
