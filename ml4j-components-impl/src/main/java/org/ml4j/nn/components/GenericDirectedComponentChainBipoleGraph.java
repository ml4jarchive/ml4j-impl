package org.ml4j.nn.components;

public abstract class GenericDirectedComponentChainBipoleGraph<I, CH extends DirectedComponentChain<I, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<I, ?>> extends DirectedComponentsBipoleGraphImpl<I, CH, DirectedComponentChainBatch<I, CH, CHA, DirectedComponentBatchActivation<I, CHA>>, CHA, DirectedComponentBatchActivation<I, CHA>, DirectedComponentsBipoleGraphActivation<I>,  DirectedComponentsContext, DirectedComponentsContext> {


	public GenericDirectedComponentChainBipoleGraph(
			DirectedComponentChainBatch<I, CH, CHA, DirectedComponentBatchActivation<I, CHA>> edges, PathCombinationStrategy pathCombinationStrategy) {
		super(edges, pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	protected DirectedComponentsBipoleGraphActivation<I> createActivation(
			OneToManyDirectedComponentActivation<I> oneToManyActivation,
			DirectedComponentBatchActivation<I, CHA> batchActivation,
			ManyToOneDirectedComponentActivation<I> manyToOneAct) {
		return  new DirectedComponentsBipoleGraphActivationImpl<>(oneToManyActivation, batchActivation, manyToOneAct);

	}

	
}
