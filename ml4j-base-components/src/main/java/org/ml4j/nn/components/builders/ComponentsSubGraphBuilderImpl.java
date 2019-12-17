package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsSubGraphBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsNestedGraphBuilderImpl<P, ComponentsSubGraphBuilder<P>> implements ComponentsSubGraphBuilder<P>, PathEnder<P, ComponentsSubGraphBuilder<P>> {
	
	public ComponentsSubGraphBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState, 
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
	}

	@Override
	public PathEnder<P, ComponentsSubGraphBuilder<P>> endPath() {
		return this;
	}

	@Override
	public ComponentsSubGraphBuilder<P> getBuilder() {
		return this;
	}

	@Override
	protected ComponentsSubGraphBuilder<P> createNewNestedGraphBuilder() {
		return new ComponentsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
