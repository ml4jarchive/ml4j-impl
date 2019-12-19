package org.ml4j.nn.components.onetoone.base;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DefaultDirectedComponentChain.
 * 
 * Encapsulates a sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultDirectedComponentChainBase implements DefaultDirectedComponentChain {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBase.class);
	
	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents;

	public DefaultDirectedComponentChainBase(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		this.sequentialComponents = sequentialComponents;
	}

	protected <X, A> A forwardPropagate(NeuronsActivation input, DefaultChainableDirectedComponent<? extends A, X> component, int componentIndex, DirectedComponentsContext context) {
		return component.forwardPropagate(input, component.getContext(context, componentIndex));
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN;
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return context;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return sequentialComponents.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		return sequentialComponents;
	}

}
