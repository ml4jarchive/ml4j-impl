package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class DefaultDirectedComponentChainBipoleGraphImpl<CH extends DirectedComponentChain<NeuronsActivation, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<NeuronsActivation, ?>, B extends DirectedComponentChainBatch<NeuronsActivation, CH, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>>>
	 extends DirectedComponentsBipoleGraphImpl<NeuronsActivation, CH, B, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>, DefaultDirectedComponentBipoleGraphActivation,  DirectedComponentsContext, DirectedComponentsContext> {
	// GenericDirectedComponentChainBipoleGraph<NeuronsActivation, DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> implements DefaultDirectedComponentBipoleGraph {

	public DefaultDirectedComponentChainBipoleGraphImpl(DirectedComponentFactory directedComponentFactory,
			B edges,
			PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;



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

	
}
