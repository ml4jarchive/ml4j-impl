package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.base.Base3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Components3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class Components3DGraphBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> extends Base3DGraphBuilderImpl<C, D> 
implements Components3DGraphBuilder<C, D> {
	
	public Components3DGraphBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory,
			Base3DGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> withParallelPaths() {
		addAxonsIfApplicable();
		return new Components3DParallelPathsBuilderImpl<>(directedAxonsComponentFactory, this::get3DBuilder);
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<C, D> withSkipConnection() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return get3DBuilder();
	}

	@Override
	public Components3DGraphBuilder<C, D> withComponents(Components3DGraphBuilder<?, ?> builder) {
		addComponents(builder.getComponents());
		return this;
	}
	
	
}
