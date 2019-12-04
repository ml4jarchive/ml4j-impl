package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilderImpl;

public class AxonsParallelPathsBuilderImpl<C extends AxonsBuilder> implements ParallelPathsBuilder<AxonsSubGraphBuilder<C>> {

	private DirectedAxonsComponentFactory directedAxonsComponentFactory;
	private Supplier<C> previousSupplier;
	private AxonsSubGraphBuilder<C> currentPath;
	
	public AxonsParallelPathsBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Supplier<C> previousSupplier) {
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public AxonsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new AxonsSubGraphBuilderImpl<>(previousSupplier, directedAxonsComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
