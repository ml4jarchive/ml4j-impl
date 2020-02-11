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
import org.ml4j.nn.components.builders.Components3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class Components3DParallelPathsBuilderImpl<C extends Components3DGraphBuilder<C, D, T>, D extends ComponentsGraphBuilder<D, T>, T extends NeuralComponent<?>>
		implements ParallelPathsBuilder<Components3DSubGraphBuilder<C, D, T>> {

	private NeuralComponentFactory<T> directedComponentFactory;
	private Supplier<C> previousSupplier;
	private Components3DSubGraphBuilder<C, D, T> currentPath;
	private DirectedComponentsContext directedComponentsContext;

	public Components3DParallelPathsBuilderImpl(NeuralComponentFactory<T> directedComponentFactory,
			Supplier<C> previousSupplier, DirectedComponentsContext directedComponentsContext) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public Components3DSubGraphBuilder<C, D, T> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath = new Components3DSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory,
				previousSupplier.get().getBuilderState(), directedComponentsContext, new ArrayList<>());
		return currentPath;
	}
}
