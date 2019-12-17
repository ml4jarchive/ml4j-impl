package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class Axons3DParallelPathsBuilderImpl<C extends Axons3DGraphBuilder<C, D>, D extends AxonsGraphBuilder<D>> implements ParallelPathsBuilder<Axons3DSubGraphBuilder<C, D>> {

	private DirectedComponentFactory directedComponentFactory;
	private Supplier<C> previousSupplier;
	private Supplier<D> previousNon3DSupplier;

	private Axons3DSubGraphBuilder<C, D> currentPath;
	
	public Axons3DParallelPathsBuilderImpl(DirectedComponentFactory directedComponentFactory, Supplier<C> previousSupplier, Supplier<D> previousNon3DSupplier) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
		this.previousNon3DSupplier = previousNon3DSupplier;
	}
	
	@Override
	public Axons3DSubGraphBuilder<C, D> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new Axons3DSubGraphBuilderImpl<>(previousSupplier, previousNon3DSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
