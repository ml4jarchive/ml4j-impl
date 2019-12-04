package org.ml4j.nn.components.builders.skipconnection;

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ComponentsGraphSkipConnectionBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsNestedGraphBuilderImpl<P, ComponentsGraphSkipConnectionBuilder<P>> implements ComponentsGraphSkipConnectionBuilder<P> {

	public ComponentsGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory, BaseGraphBuilderState builderState, 
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(parentGraph, directedAxonsComponentFactory, builderState, components);
	}

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parentGraph.get();
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<P> getBuilder() {
		return this;
	}

	@Override
	protected ComponentsGraphSkipConnectionBuilder<P> createNewNestedGraphBuilder() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(parentGraph, directedAxonsComponentFactory, initialBuilderState, components);
	}
}
