package org.ml4j.nn.components.builders.skipconnection;

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsGraphSkipConnectionBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsNestedGraphBuilderImpl<P, ComponentsGraphSkipConnectionBuilder<P>> implements ComponentsGraphSkipConnectionBuilder<P> {

	public ComponentsGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState, 
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
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
		return new ComponentsGraphSkipConnectionBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, components);
	}
}
