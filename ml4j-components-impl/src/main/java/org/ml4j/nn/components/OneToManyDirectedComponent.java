package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

public abstract class OneToManyDirectedComponent<I, C> implements DirectedComponent<I, OneToManyDirectedComponentActivation<I>, C> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents;
	
	public OneToManyDirectedComponent(List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents) {
		this.targetComponents = targetComponents;
	}
	
	@Override
	public OneToManyDirectedComponentActivation<I> forwardPropagate(I input,
			C synapsesContext) {
		List<I> acts = new ArrayList<>();
		for (int i = 0; i < targetComponents.size(); i++) {
			acts.add(input);
		}
		
		return createActivation(acts, synapsesContext);
	}

	protected abstract OneToManyDirectedComponentActivation<I> createActivation(List<I> acts, C synapsesContext);

}
