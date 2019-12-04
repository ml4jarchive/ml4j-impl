package org.ml4j.nn.components.builders.skipconnection;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.neurons.NeuronsActivation;

public class Components3DGraphSkipConnectionBuilderImpl<P extends Components3DGraphBuilder<P, Q>, Q extends ComponentsGraphBuilder<Q>>
		extends ComponentsNested3DGraphBuilderImpl<P, Components3DGraphSkipConnectionBuilder<P, Q>, ComponentsGraphSkipConnectionBuilder<Q>>
		implements Components3DGraphSkipConnectionBuilder<P, Q>, PathEnder<P, Components3DGraphSkipConnectionBuilder<P, Q>>{

	private ComponentsGraphSkipConnectionBuilder<Q> builder;
	
	public Components3DGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			Base3DGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(parentGraph, directedAxonsComponentFactory, builderState, components);
	}
	

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parent3DGraph.get();
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			builder =  new ComponentsGraphSkipConnectionBuilderImpl<>(() -> parent3DGraph.get().getBuilder(), directedAxonsComponentFactory,
					builderState.getNon3DBuilderState(), getComponents());
			return builder;
		}
	}


	@Override
	protected Components3DGraphSkipConnectionBuilder<P, Q> createNewNestedGraphBuilder() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, directedAxonsComponentFactory, initialBuilderState, new ArrayList<>());
	}
	
	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q>, ComponentsGraphSkipConnectionBuilder<Q>> withComponents(
			Components3DGraphBuilder<?, ?> builder) {
		addComponents(builder.getComponents());
		return this;
	}

	
}
