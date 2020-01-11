/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.builders.axonsgraph;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Axons3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.skipconnection.Axons3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class Axons3DGraphBuilderImpl<C extends Axons3DGraphBuilder<C, D, T>, D extends AxonsGraphBuilder<D, T>, T extends NeuralComponent> extends BaseNested3DGraphBuilderImpl<C, C, D, T> 
implements Axons3DGraphBuilder<C, D, T>, PathEnder<C, C> {
	
	private Supplier<D> parentNon3DGraph;
	private C currentPath;
	
	public Axons3DGraphBuilderImpl(Supplier<C> parent3DGraph, Supplier<D> parentNon3DGraph, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parent3DGraph, directedComponentFactory, builderState, directedComponentsContext, components);
		this.parentNon3DGraph = parentNon3DGraph;
	}

	@Override
	public ParallelPathsBuilder<Axons3DSubGraphBuilder<C, D, T>> withParallelPaths() {
		addAxonsIfApplicable();
		return new Axons3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, this::getBuilder, directedComponentsContext);
	}

	@Override
	public Axons3DGraphSkipConnectionBuilder<C, D, T> withSkipConnection() {
		return new Axons3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
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
