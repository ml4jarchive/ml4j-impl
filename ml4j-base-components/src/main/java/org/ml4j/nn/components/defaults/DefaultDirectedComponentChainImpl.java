package org.ml4j.nn.components.defaults;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivationImpl;
import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainImpl extends DirectedComponentChainBaseImpl<NeuronsActivation, DefaultChainableDirectedComponent<?, ?>, DefaultChainableDirectedComponentActivation, DefaultDirectedComponentChainActivation> implements DefaultDirectedComponentChain {


	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	public DefaultDirectedComponentChainImpl(
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(components);
	}

	@Override
	protected DefaultDirectedComponentChainActivation createChainActivation(
			List<DefaultChainableDirectedComponentActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DefaultDirectedComponentChainActivationImpl(componentActivations, inFlightInput);
	}

	@Override
	public DefaultDirectedComponentChain dup() {
		return new DefaultDirectedComponentChainImpl(components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return components.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN;
	}
}
