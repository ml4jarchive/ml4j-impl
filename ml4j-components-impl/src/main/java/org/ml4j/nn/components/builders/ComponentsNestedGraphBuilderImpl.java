package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ComponentsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class ComponentsNestedGraphBuilderImpl<P extends ComponentsContainer<Neurons>, C extends AxonsBuilder> extends BaseNestedGraphBuilderImpl<P, C> implements ComponentsGraphBuilder<C>{
	

	public ComponentsNestedGraphBuilderImpl(Supplier<P> parentGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			BaseGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(parentGraph, directedAxonsComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> withParallelPaths() {
		addAxonsIfApplicable();
		return new ComponentsParallelPathsBuilderImpl<>(directedAxonsComponentFactory, this::getBuilder);
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<C> withSkipConnection() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return getBuilder();
	}
}
