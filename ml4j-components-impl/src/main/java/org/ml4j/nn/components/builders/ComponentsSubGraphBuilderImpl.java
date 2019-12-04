package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ComponentsSubGraphBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsNestedGraphBuilderImpl<P, ComponentsSubGraphBuilder<P>> implements ComponentsSubGraphBuilder<P>, PathEnder<P, ComponentsSubGraphBuilder<P>> {
	
	public ComponentsSubGraphBuilderImpl(Supplier<P> parentGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory, BaseGraphBuilderState builderState, 
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(parentGraph, directedAxonsComponentFactory, builderState, components);
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
		return new ComponentsSubGraphBuilderImpl<>(parentGraph, directedAxonsComponentFactory, initialBuilderState, new ArrayList<>());
	}
}
