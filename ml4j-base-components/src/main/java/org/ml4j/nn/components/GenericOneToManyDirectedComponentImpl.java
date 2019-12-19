package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntSupplier;

public abstract class GenericOneToManyDirectedComponentImpl<I, C, A extends GenericOneToManyDirectedComponentActivation<I>> implements GenericOneToManyDirectedComponent<I, C, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected IntSupplier targetComponentsCount;
	
	public GenericOneToManyDirectedComponentImpl(IntSupplier targetComponentsCount) {
		this.targetComponentsCount = targetComponentsCount;
	}
	
	@Override
	public A forwardPropagate(I input,
			C synapsesContext) {
		List<I> acts = new ArrayList<>();
		for (int i = 0; i < targetComponentsCount.getAsInt(); i++) {
			acts.add(input);
		}
		
		return createActivation(acts, synapsesContext);
	}

	protected abstract A createActivation(List<I> acts, C synapsesContext);

}
