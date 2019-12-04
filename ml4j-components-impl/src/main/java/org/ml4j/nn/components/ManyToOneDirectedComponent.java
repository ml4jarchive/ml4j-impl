package org.ml4j.nn.components;

import java.util.List;

public abstract class ManyToOneDirectedComponent<I, C> implements DirectedComponent<List<I>, ManyToOneDirectedComponentActivation<I>, C> {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
		

	@Override
	public ManyToOneDirectedComponentActivation<I> forwardPropagate(List<I> input,
			C context) {
		I combinedInput = getCombinedOutput(input, context);
		return createActivation(combinedInput, input);
	}
	
	protected  ManyToOneDirectedComponentActivation<I> createActivation(I combinedInput, List<I> input) {
		return new ManyToOneDirectedComponentActivation<>(combinedInput, input.size());
	}
	
	protected abstract I getCombinedOutput(List<I> input, C context);

}
