package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class AxonsParallelPathsBuilderImpl<C extends AxonsBuilder> implements ParallelPathsBuilder<AxonsSubGraphBuilder<C>> {

	private DirectedComponentFactory directedComponentFactory;
	private Supplier<C> previousSupplier;
	private AxonsSubGraphBuilder<C> currentPath;
	
	public AxonsParallelPathsBuilderImpl(DirectedComponentFactory directedComponentFactory, Supplier<C> previousSupplier) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public AxonsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new AxonsSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
