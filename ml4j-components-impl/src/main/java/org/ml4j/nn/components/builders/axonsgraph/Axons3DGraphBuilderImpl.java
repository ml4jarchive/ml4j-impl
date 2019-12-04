package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class Axons3DGraphBuilderImpl<C extends Axons3DGraphBuilder<C, D>, D extends AxonsGraphBuilder<D>> extends BaseNested3DGraphBuilderImpl<C, C, D> 
implements Axons3DGraphBuilder<C, D>, PathEnder<C, C> {
	
	private Supplier<D> parentNon3DGraph;
	private C currentPath;
	
	public Axons3DGraphBuilderImpl(Supplier<C> parent3DGraph, Supplier<D> parentNon3DGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			Base3DGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(parent3DGraph, directedAxonsComponentFactory, builderState, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}

	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<C, D>> withParallelPaths() {
		addAxonsIfApplicable();
		return new Axons3DParallelPathsBuilderImpl<>(directedAxonsComponentFactory, this::get3DBuilder, this::getBuilder);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<C, D> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C get3DBuilder() {
		return parent3DGraph.get();	
	}

	@Override
	public D getBuilder() {
		return parentNon3DGraph.get();
	}
	
	@Override
	public C withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		completeNestedGraph(false);
		currentPath = createNewNestedGraphBuilder();
		return currentPath;
	}

	@Override
	protected C createNewNestedGraphBuilder() {
		throw new UnsupportedOperationException("Not supported");
	}

}
