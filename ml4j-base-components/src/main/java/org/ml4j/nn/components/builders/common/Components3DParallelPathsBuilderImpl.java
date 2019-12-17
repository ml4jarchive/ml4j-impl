package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.Components3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class Components3DParallelPathsBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> implements ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> {

	private DirectedComponentFactory directedComponentFactory;
	private Supplier<C> previousSupplier;
	private Components3DSubGraphBuilder<C, D> currentPath;
	
	public Components3DParallelPathsBuilderImpl(DirectedComponentFactory directedComponentFactory, Supplier<C> previousSupplier) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public Components3DSubGraphBuilder<C, D> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath = new Components3DSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
