package org.ml4j.nn.components;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;

public abstract class GenericDirectedComponentChainBipoleGraph<I, CH extends DirectedComponentChain<I, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<I, ?>> extends DirectedComponentsBipoleGraphImpl<I, CH, DirectedComponentChainBatch<I, CH, CHA, DirectedComponentBatchActivation<I, CHA>>, CHA, DirectedComponentBatchActivation<I, CHA>, DirectedComponentBipoleGraphActivation<I>,  DirectedComponentsContext, DirectedComponentsContext> {


	public GenericDirectedComponentChainBipoleGraph(DirectedComponentFactory directedComponentFactory, 
			DirectedComponentChainBatch<I, CH, CHA, DirectedComponentBatchActivation<I, CHA>> edges, PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;


	
}
