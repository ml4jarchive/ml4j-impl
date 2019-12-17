package org.ml4j.nn.components.defaults;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultChainableDirectedComponentAdapter<C> implements DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, C> {

	/**
	 * Defaul serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, C> delegated;
	
	public DefaultChainableDirectedComponentAdapter(ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, C> delegated) {
		this.delegated = delegated;
	}

	@Override
	public DefaultChainableDirectedComponentActivation forwardPropagate(NeuronsActivation input, C context) {
		ChainableDirectedComponentActivation<NeuronsActivation> delegatedActivation = delegated.forwardPropagate(input, context);
		return new DefaultChainableDirectedComponentActivationAdapter(delegatedActivation);
	}

	@Override
	public C getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return delegated.getContext(directedComponentsContext, componentIndex);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return delegated.decompose().stream().map(c -> createChainableDirectedComponent(c)).collect(Collectors.toList());
	}

	private DefaultChainableDirectedComponent<?, ?>  createChainableDirectedComponent(
			ChainableDirectedComponent<NeuronsActivation, ? extends 
					ChainableDirectedComponentActivation<NeuronsActivation>, ?> c) {
		return new DefaultChainableDirectedComponentAdapter<>(c);
	}

	@Override
	public DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, C> dup() {
		return new DefaultChainableDirectedComponentAdapter<C>(delegated.dup());
	}
	
}
