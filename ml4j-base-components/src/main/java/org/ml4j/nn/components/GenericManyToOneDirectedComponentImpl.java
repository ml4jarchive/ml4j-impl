package org.ml4j.nn.components;

import java.util.List;

public abstract class GenericManyToOneDirectedComponentImpl<I, C, A extends GenericManyToOneDirectedComponentActivation<I>> implements DirectedComponent<List<I>, A, C> {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
		

	@Override
	public A forwardPropagate(List<I> input,
			C context) {
		I combinedInput = getCombinedOutput(input, context);
		return createActivation(combinedInput, input);
	}
	
	protected abstract A createActivation(I combinedInput, List<I> input);
	
	protected abstract I getCombinedOutput(List<I> input, C context);

}
