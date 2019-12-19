package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public abstract class ComponentsGraphBuilderImpl<C extends AxonsBuilder> extends BaseGraphBuilderImpl<C> implements ComponentsGraphBuilder<C>{
	
	
	public ComponentsGraphBuilderImpl(DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
			super(directedComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> withParallelPaths() {
		addAxonsIfApplicable();
		return new ComponentsParallelPathsBuilderImpl<>(directedComponentFactory, this::getBuilder);
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<C> withSkipConnection() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return getBuilder();
	}
}
