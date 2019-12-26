package org.ml4j.nn.components.onetoone.base;



import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.generic.DirectedComponentChain;
import org.ml4j.nn.components.generic.DirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.Neurons;
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
public abstract class DefaultDirectedComponentChainBaseParent<L extends DefaultChainableDirectedComponent<? extends A, ?>, 
	A extends DefaultChainableDirectedComponentActivation, CH extends DirectedComponentChainActivation<NeuronsActivation, A>> implements 
	DirectedComponentChain<NeuronsActivation, L, A, CH> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBaseParent.class);
	
	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<L> sequentialComponents;

	public DefaultDirectedComponentChainBaseParent(List<L> sequentialComponents) {
		this.sequentialComponents = sequentialComponents;
	}

	protected <X, Y> Y forwardPropagate(NeuronsActivation input, DefaultChainableDirectedComponent<? extends Y, X> component, int componentIndex, DirectedComponentsContext context) {
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
	public List<L> getComponents() {
		return sequentialComponents;
	}

	public Neurons getInputNeurons() {
		return sequentialComponents.get(0).getInputNeurons();
	}

	public Neurons getOutputNeurons() {
		return sequentialComponents.get(sequentialComponents.size() - 1).getOutputNeurons();
	}
	
	

}
