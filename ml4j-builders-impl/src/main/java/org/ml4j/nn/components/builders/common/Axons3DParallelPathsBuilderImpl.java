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
package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.function.Supplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.Axons3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class Axons3DParallelPathsBuilderImpl<C extends Axons3DGraphBuilder<C, D, T>, D extends AxonsGraphBuilder<D, T>, T extends NeuralComponent<?>>
		implements ParallelPathsBuilder<Axons3DSubGraphBuilder<C, D, T>> {

	private NeuralComponentFactory<T> directedComponentFactory;
	private Supplier<C> previousSupplier;
	private Supplier<D> previousNon3DSupplier;
	private DirectedComponentsContext directedComponentsContext;

	private Axons3DSubGraphBuilder<C, D, T> currentPath;

	public Axons3DParallelPathsBuilderImpl(NeuralComponentFactory<T> directedComponentFactory,
			Supplier<C> previousSupplier, Supplier<D> previousNon3DSupplier,
			DirectedComponentsContext directedComponentsContext) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
		this.previousNon3DSupplier = previousNon3DSupplier;
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public Axons3DSubGraphBuilder<C, D, T> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath = new Axons3DSubGraphBuilderImpl<>(previousSupplier, previousNon3DSupplier,
				directedComponentFactory, previousSupplier.get().getBuilderState(), directedComponentsContext,
				new ArrayList<>());
		return currentPath;
	}
}
