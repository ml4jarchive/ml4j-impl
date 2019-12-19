package org.ml4j.nn.components;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;


public abstract class DirectedComponentsBipoleGraphImpl<I, L extends ChainableDirectedComponent<I, A, C>, B extends DirectedComponentBatch<I, L , Y, A, C, C2>, A extends ChainableDirectedComponentActivation<I>, Y extends DirectedComponentBatchActivation<I, A>, Z extends DirectedComponentBipoleGraphActivation<I>, C, C2> implements DirectedComponentBipoleGraph<I,  C2, Z, B> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private GenericOneToManyDirectedComponent<I, C2, ?> inputLink;
	private GenericManyToOneDirectedComponent<I, C2, ?> outputLink;
	protected PathCombinationStrategy pathCombinationStrategy;
	protected DirectedComponentFactory directedComponentFactory;
	protected B edges;

	public DirectedComponentsBipoleGraphImpl(DirectedComponentFactory directedComponentFactory, B edges, PathCombinationStrategy pathCombinationStrategy) {
		this.pathCombinationStrategy = pathCombinationStrategy;
		this.inputLink = createOneToManyDirectedComponent(directedComponentFactory, edges.getComponents());
		this.outputLink = createManyToOneDirectedComponent(directedComponentFactory);
		this.directedComponentFactory = directedComponentFactory;
		this.edges = edges;
	}
	
	protected abstract GenericOneToManyDirectedComponent<I, C2, ?> createOneToManyDirectedComponent(DirectedComponentFactory directedComponentFactory, List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, C>> targetComponents);
	
	protected abstract GenericManyToOneDirectedComponent<I, C2, ?> createManyToOneDirectedComponent(DirectedComponentFactory directedComponentFactory);

	public B getEdges() {
		return edges;
	}

	@Override
	public Z forwardPropagate(I input, C2 batchContext) {
		GenericOneToManyDirectedComponentActivation<I> oneToManyActivation = inputLink.forwardPropagate(input, batchContext);
		Y batchActivation = edges.forwardPropagate(oneToManyActivation.getOutput(), batchContext);
		GenericManyToOneDirectedComponentActivation<I> manyToOneAct = outputLink.forwardPropagate(batchActivation.getOutput(), batchContext);
		return createActivation(oneToManyActivation, batchActivation, manyToOneAct);
	}
	
	protected abstract Z createActivation(GenericOneToManyDirectedComponentActivation<I> oneToManyActivation, Y batchActivation, 
	GenericManyToOneDirectedComponentActivation<I> manyToOneAct);

	//@Override
	//public DirectedComponentsBipoleGraph<I, E, C> dup() {
	//	return new DirectedComponentsBipoleGraph<>(edges.dup());
	//}

	@Override
	public List<? extends ChainableDirectedComponent<I, ? extends ChainableDirectedComponentActivation<I>, ?>> decompose() {
		return Arrays.asList(this);
	}
}
