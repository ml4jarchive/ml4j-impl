package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

public abstract class GenericOneToManyDirectedComponentImpl<I, C, A extends GenericOneToManyDirectedComponentActivation<I>> implements GenericOneToManyDirectedComponent<I, C, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents;
	
	public GenericOneToManyDirectedComponentImpl(List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents) {
		this.targetComponents = targetComponents;
	}
	
	@Override
	public A forwardPropagate(I input,
			C synapsesContext) {
		List<I> acts = new ArrayList<>();
		for (int i = 0; i < targetComponents.size(); i++) {
			acts.add(input);
		}
		
		return createActivation(acts, synapsesContext);
	}

	protected abstract A createActivation(List<I> acts, C synapsesContext);

}
