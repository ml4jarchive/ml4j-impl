package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.ComponentsSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;

public class ComponentsParallelPathsBuilderImpl<C extends AxonsBuilder> implements ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> {

	private DirectedAxonsComponentFactory directedAxonsComponentFactory;
	private Supplier<C> previousSupplier;
	private ComponentsSubGraphBuilder<C> currentPath;
	
	public ComponentsParallelPathsBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Supplier<C> previousSupplier) {
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public ComponentsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new ComponentsSubGraphBuilderImpl<>(previousSupplier, directedAxonsComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
