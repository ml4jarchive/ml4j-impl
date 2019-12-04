package org.ml4j.nn.components;

import java.util.Arrays;
import java.util.List;


public abstract class DirectedComponentsBipoleGraphImpl<I, L extends ChainableDirectedComponent<I, A, C>, B extends DirectedComponentBatch<I, L , Y, A, C, C2>, A extends ChainableDirectedComponentActivation<I>, Y extends DirectedComponentBatchActivation<I, A>, Z extends DirectedComponentsBipoleGraphActivation<I>, C, C2> implements DirectedComponentBipoleGraph<I,  C2, Z, B> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private OneToManyDirectedComponent<I, C2> inputLink;
	private ManyToOneDirectedComponent<I, C2> outputLink;
	protected PathCombinationStrategy pathCombinationStrategy;
	protected B edges;

	public DirectedComponentsBipoleGraphImpl(B edges, PathCombinationStrategy pathCombinationStrategy) {
		this.pathCombinationStrategy = pathCombinationStrategy;
		this.inputLink = createOneToManyDirectedComponent(edges.getComponents());
		this.outputLink = createManyToOneDirectedComponent();
		this.edges = edges;
	}
	
	protected abstract OneToManyDirectedComponent<I, C2> createOneToManyDirectedComponent(List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents);
	
	protected abstract ManyToOneDirectedComponent<I, C2> createManyToOneDirectedComponent();

	public B getEdges() {
		return edges;
	}

	@Override
	public Z forwardPropagate(I input, C2 batchContext) {
		OneToManyDirectedComponentActivation<I> oneToManyActivation = inputLink.forwardPropagate(input, batchContext);
		Y batchActivation = edges.forwardPropagate(oneToManyActivation.getOutput(), batchContext);
		ManyToOneDirectedComponentActivation<I> manyToOneAct = outputLink.forwardPropagate(batchActivation.getOutput(), batchContext);
		return createActivation(oneToManyActivation, batchActivation, manyToOneAct);
	}
	
	protected abstract Z createActivation(OneToManyDirectedComponentActivation<I> oneToManyActivation, Y batchActivation, 
	ManyToOneDirectedComponentActivation<I> manyToOneAct);

	//@Override
	//public DirectedComponentsBipoleGraph<I, E, C> dup() {
	//	return new DirectedComponentsBipoleGraph<>(edges.dup());
	//}

	@Override
	public List<ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, ?>> decompose() {
		return Arrays.asList(this);
	}
}
