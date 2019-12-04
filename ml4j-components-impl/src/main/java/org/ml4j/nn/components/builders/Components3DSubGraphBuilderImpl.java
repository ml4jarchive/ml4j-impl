package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.NeuronsActivation;

public class Components3DSubGraphBuilderImpl<P extends Components3DGraphBuilder<P, Q>, Q extends ComponentsGraphBuilder<Q>>
		extends ComponentsNested3DGraphBuilderImpl<P, Components3DSubGraphBuilder<P, Q>, ComponentsSubGraphBuilder<Q>>
		implements Components3DSubGraphBuilder<P, Q>, PathEnder<P, Components3DSubGraphBuilder<P, Q>> {

	private Supplier<P> previousSupplier;
	private ComponentsSubGraphBuilder<Q> builder;
	
	public Components3DSubGraphBuilderImpl(Supplier<P> previousSupplier, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			Base3DGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(previousSupplier, directedAxonsComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public Components3DSubGraphBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsSubGraphBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			previousSupplier.get().addComponents(getComponents());
			builder =  new ComponentsSubGraphBuilderImpl<>(() -> previousSupplier.get().getBuilder(), directedAxonsComponentFactory,
					builderState.getNon3DBuilderState(), new ArrayList<>());
			return builder;
		}
	}
	
	@Override
	public Components3DSubGraphBuilder<P, Q> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}

	@Override
	public Components3DSubGraphBuilder<P, Q> withActivationFunction(
			DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return this;
	}

	@Override
	protected Components3DSubGraphBuilder<P, Q> createNewNestedGraphBuilder() {
		// FIX
		return new Components3DSubGraphBuilderImpl<>(previousSupplier,
				directedAxonsComponentFactory, initialBuilderState, new ArrayList<>());
	}

	@Override
	public PathEnder<P, Components3DSubGraphBuilder<P, Q>> endPath() {
		completeNestedGraph(false);
		return this;
	}

	@Override
	public Components3DGraphBuilder<Components3DSubGraphBuilder<P, Q>, ComponentsSubGraphBuilder<Q>> withComponents(
			Components3DGraphBuilder<?, ?> builder) {
		addComponents(builder.getComponents());
		return this;
	}

}
