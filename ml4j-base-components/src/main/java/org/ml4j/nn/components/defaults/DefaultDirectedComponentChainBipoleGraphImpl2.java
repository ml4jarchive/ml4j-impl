package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentBipoleGraph;
import org.ml4j.nn.components.DirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericManyToOneDirectedComponent;
import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.GenericOneToManyDirectedComponent;
import org.ml4j.nn.components.GenericOneToManyDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBipoleGraphImpl2
		extends DefaultDirectedComponentChainBipoleGraphImpl<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation, DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation>>
	implements DefaultDirectedComponentBipoleGraph {

	public DefaultDirectedComponentChainBipoleGraphImpl2(DirectedComponentFactory directedComponentFactory,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> edges,
			PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext arg0, int arg1) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DirectedComponentChainBatch<NeuronsActivation, DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation, DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation>> getEdges() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
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
	protected DefaultDirectedComponentBipoleGraphActivation createActivation(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation> batchActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		// TODO Auto-generated method stub
		return null;
	}

	

}
