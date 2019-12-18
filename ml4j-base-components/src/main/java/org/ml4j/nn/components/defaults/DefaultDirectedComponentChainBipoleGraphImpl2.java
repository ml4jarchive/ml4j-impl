package org.ml4j.nn.components.defaults;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.DefaultDirectedComponentsBipoleGraphActivationImpl;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.GenericOneToManyDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBipoleGraphImpl2
		extends DefaultDirectedComponentChainBipoleGraphImpl<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation, DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation>>
	implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainBipoleGraphImpl2(DirectedComponentFactory directedComponentFactory,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> edges,
			PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
		return new DefaultDirectedComponentChainBipoleGraphImpl2(directedComponentFactory, edges.dup(),pathCombinationStrategy);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int arg1) {
		return context;
	}

	@Override
	protected DefaultDirectedComponentBipoleGraphActivation createActivation(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation> batchActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		return new DefaultDirectedComponentsBipoleGraphActivationImpl(oneToManyActivation, batchActivation, manyToOneAct);
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN_GRAPH;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}
}
