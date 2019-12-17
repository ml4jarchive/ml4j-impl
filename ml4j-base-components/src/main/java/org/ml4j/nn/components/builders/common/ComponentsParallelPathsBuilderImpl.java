package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.ComponentsSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class ComponentsParallelPathsBuilderImpl<C extends AxonsBuilder> implements ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> {

	private DirectedComponentFactory directedComponentFactory;
	private Supplier<C> previousSupplier;
	private ComponentsSubGraphBuilder<C> currentPath;
	
	public ComponentsParallelPathsBuilderImpl(DirectedComponentFactory directedComponentFactory, Supplier<C> previousSupplier) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public ComponentsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new ComponentsSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
