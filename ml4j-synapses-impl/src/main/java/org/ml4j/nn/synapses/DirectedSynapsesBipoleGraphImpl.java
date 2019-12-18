package org.ml4j.nn.synapses;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentBatch;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.GenericOneToManyDirectedComponentActivation;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesBipoleGraphImpl<S extends DirectedSynapses<?, ?>> extends DirectedComponentsBipoleGraphImpl<NeuronsActivation, DirectedSynapsesChain<S>, DirectedComponentBatch<NeuronsActivation, DirectedSynapsesChain<S> , DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesChainActivation, DirectedComponentsContext, DirectedComponentsContext>, DirectedSynapsesChainActivation, DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesBipoleGraphActivation, DirectedComponentsContext, DirectedComponentsContext>
	implements DirectedSynapsesBipoleGraph<S> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedSynapsesBipoleGraphImpl(DirectedComponentFactory directedComponentFactory,
			DirectedComponentBatch<NeuronsActivation, DirectedSynapsesChain<S>, DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesChainActivation, DirectedComponentsContext, 
			DirectedComponentsContext> edges, PathCombinationStrategy pathCombinationStrategy) {
		super(directedComponentFactory, edges, pathCombinationStrategy);
	}

	@Override
	protected DirectedSynapsesBipoleGraphActivation createActivation(
			GenericOneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation> batchActivation,
			GenericManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		return new DirectedSynapsesBipoleGraphActivationImpl(oneToManyActivation, batchActivation, manyToOneAct);

	}

	@Override
	protected ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(DirectedComponentFactory directedComponentFactory) {
		return directedComponentFactory.createManyToOneDirectedComponent(pathCombinationStrategy);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext,
			int componentIndex) {
		return directedComponentsContext;
	}

	@Override
	protected OneToManyDirectedComponent<?> createOneToManyDirectedComponent(DirectedComponentFactory directedComponentFactory,
			List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		return directedComponentFactory.createOneToManyDirectedComponent(targetComponents);
	}

	@Override
	public DirectedSynapsesBipoleGraph<S> dup() {
		return new DirectedSynapsesBipoleGraphImpl<S>(this.directedComponentFactory, edges.dup(), pathCombinationStrategy);
		}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.SYNAPSES_GRAPH;
	}	
}
