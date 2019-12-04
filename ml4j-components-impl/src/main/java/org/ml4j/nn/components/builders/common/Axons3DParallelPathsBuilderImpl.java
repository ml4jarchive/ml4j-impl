package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphBuilder;

public class Axons3DParallelPathsBuilderImpl<C extends Axons3DGraphBuilder<C, D>, D extends AxonsGraphBuilder<D>> implements ParallelPathsBuilder<Axons3DSubGraphBuilder<C, D>> {

	private DirectedAxonsComponentFactory directedAxonsComponentFactory;
	private Supplier<C> previousSupplier;
	private Supplier<D> previousNon3DSupplier;

	private Axons3DSubGraphBuilder<C, D> currentPath;
	
	public Axons3DParallelPathsBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Supplier<C> previousSupplier, Supplier<D> previousNon3DSupplier) {
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		this.previousSupplier = previousSupplier;
		this.previousNon3DSupplier = previousNon3DSupplier;
	}
	
	@Override
	public Axons3DSubGraphBuilder<C, D> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new Axons3DSubGraphBuilderImpl<>(previousSupplier, previousNon3DSupplier, directedAxonsComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
