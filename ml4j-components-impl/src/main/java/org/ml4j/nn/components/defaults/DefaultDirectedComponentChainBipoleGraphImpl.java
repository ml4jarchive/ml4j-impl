package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBipoleGraphImpl<CH extends DirectedComponentChain<NeuronsActivation, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<NeuronsActivation, ?>> extends GenericDirectedComponentChainBipoleGraph<NeuronsActivation, CH, CHA> {
	
	public DefaultDirectedComponentChainBipoleGraphImpl(
			DirectedComponentChainBatch<NeuronsActivation, CH, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>> edges, PathCombinationStrategy pathCombinationStrategy) {
		super(edges, pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	protected ManyToOneDirectedComponent<NeuronsActivation, DirectedComponentsContext> createManyToOneDirectedComponent() {
		if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT) {
			return new DefaultManyToOneFilterConcatDirectedComponent();
		} else {
			return new DefaultManyToOneAdditionDirectedComponent();
		}
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext,
			int componentIndex) {
		return directedComponentsContext;
	}
	
	public String toString() {
		return getClass().getName();
	}

	@Override
	protected OneToManyDirectedComponent<NeuronsActivation, DirectedComponentsContext> createOneToManyDirectedComponent(
			List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		return new DefaultOneToManyDirectedComponent(targetComponents);
	}
	
}
