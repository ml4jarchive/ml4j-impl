package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultDirectedComponentBatch;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.GenericManyToOneDirectedComponent;
import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.GenericOneToManyDirectedComponent;
import org.ml4j.nn.components.GenericOneToManyDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class DefaultDirectedComponentChainBipoleGraphImpl<CH extends DirectedComponentChain<NeuronsActivation, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<NeuronsActivation, ?>, B extends DirectedComponentChainBatch<NeuronsActivation, CH, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>>>
	 extends DirectedComponentsBipoleGraphImpl<NeuronsActivation, CH, B, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>, DefaultDirectedComponentBipoleGraphActivation,  DirectedComponentsContext, DirectedComponentsContext> {
	// GenericDirectedComponentChainBipoleGraph<NeuronsActivation, DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> implements DefaultDirectedComponentBipoleGraph {

	public DefaultDirectedComponentChainBipoleGraphImpl(DirectedComponentFactory directedComponentFactory,
			DefaultDirectedComponentChainBatch<CH, CHA> edges,
			PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/*
	@Override
	public DefaultDirectedComponentBatch<DirectedComponentsContext, DirectedComponentsContext> getEdges() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext arg0, int arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation arg0,
			DirectedComponentsContext arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DirectedComponentChainBatch<NeuronsActivation, DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation, DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation>> getEdges() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected GenericOneToManyDirectedComponent<NeuronsActivation, DirectedComponentsContext, ?> createOneToManyDirectedComponent(
			DirectedComponentFactory directedComponentFactory,
			List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected GenericManyToOneDirectedComponent<NeuronsActivation, DirectedComponentsContext, ?> createManyToOneDirectedComponent(
			DirectedComponentFactory directedComponentFactory) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected DirectedComponentBipoleGraphActivation<NeuronsActivation> createActivation(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation> batchActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		// TODO Auto-generated method stub
		return null;
	}
	
	*/

	/*
	@Override
	protected ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			DirectedComponentFactory directedComponentFactory) {
		return directedComponentFactory.createManyToOneDirectedComponent(pathCombinationStrategy);
	}

	public String toString() {
		return getClass().getName();
	}

	@Override
	protected OneToManyDirectedComponent<?> createOneToManyDirectedComponent(
			DirectedComponentFactory directedComponentFactory,
			List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		return directedComponentFactory.createOneToManyDirectedComponent(targetComponents);
	}

	@Override
	public DefaultDirectedComponentChainBipoleGraphImpl<CH, CHA> dup() {
		return new DefaultDirectedComponentChainBipoleGraphImpl<>(directedComponentFactory, edges.dup(),pathCombinationStrategy);
	}

	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation arg0,
			DirectedComponentsContext arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DirectedComponentChainBatch<NeuronsActivation, CH, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>> getEdges() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected DirectedComponentBipoleGraphActivation<NeuronsActivation> createActivation(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, CHA> batchActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		// TODO Auto-generated method stub
		return null;
	}
	*/
	
}
