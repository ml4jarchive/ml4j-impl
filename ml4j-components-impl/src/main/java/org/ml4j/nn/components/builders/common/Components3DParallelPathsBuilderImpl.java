package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.Components3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;

public class Components3DParallelPathsBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> implements ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> {

	private DirectedAxonsComponentFactory directedAxonsComponentFactory;
	private Supplier<C> previousSupplier;
	private Components3DSubGraphBuilder<C, D> currentPath;
	
	public Components3DParallelPathsBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Supplier<C> previousSupplier) {
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public Components3DSubGraphBuilder<C, D> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath = new Components3DSubGraphBuilderImpl<>(previousSupplier, directedAxonsComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
