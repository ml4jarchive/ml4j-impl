package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class DirectedComponentChainBaseImpl<I, D extends ChainableDirectedComponent<I, ? extends A, ?>, A extends ChainableDirectedComponentActivation<I>, B extends DirectedComponentChainActivation<I, A>> implements DirectedComponentChain<I, D , A, B> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	@Override
	public List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, ?>> decompose() {
		return components.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}


	protected List<D> components;
	
	public DirectedComponentChainBaseImpl(List<? extends D> components) {
		this.components = new ArrayList<>();
		this.components.addAll(components);
	}
			
	protected <X> A forwardPropagate(I input, ChainableDirectedComponent<I, ? extends A, X> component, int componentIndex, DirectedComponentsContext context) {
		return component.forwardPropagate(input, component.getContext(context, componentIndex));
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return directedComponentsContext;
	}
	
	protected void onForwardPropagation(A inFlightNeuronsComponentActivation) {
		// no-op by default
	}
	
	@Override
	public B forwardPropagate(I componentChainInput, DirectedComponentsContext context) {
		A inFlightNeuronsComponentActivation = null;
		 I inFlightInput = componentChainInput;
		 	int componentIndex = 0;
		    List<A> componentActivations = new ArrayList<>();
		    for (ChainableDirectedComponent<I, ? extends A, ?> component : components) {
	
		      inFlightNeuronsComponentActivation = forwardPropagate(inFlightInput, component, componentIndex, context);
		     
		      onForwardPropagation(inFlightNeuronsComponentActivation);
		      componentActivations.add(inFlightNeuronsComponentActivation);
		      inFlightInput = inFlightNeuronsComponentActivation.getOutput();
		      componentIndex++;
		    }
		    return createChainActivation(componentActivations, inFlightInput);
	}
	
	protected abstract B createChainActivation(List<A> componentActivations, I inFlightInput);


	@Override
	public List<D> getComponents() {
		return components;
	}

}
