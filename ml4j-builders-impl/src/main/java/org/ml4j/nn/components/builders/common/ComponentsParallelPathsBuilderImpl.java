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
import org.ml4j.nn.components.builders.ComponentsSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class ComponentsParallelPathsBuilderImpl<C extends AxonsBuilder<T>, T extends NeuralComponent> implements ParallelPathsBuilder<ComponentsSubGraphBuilder<C, T>> {

	private NeuralComponentFactory<T> directedComponentFactory;
	private Supplier<C> previousSupplier;
	private ComponentsSubGraphBuilder<C, T> currentPath;
	private DirectedComponentsContext directedComponentsContext;
	
	public ComponentsParallelPathsBuilderImpl(NeuralComponentFactory<T> directedComponentFactory, Supplier<C> previousSupplier, DirectedComponentsContext directedComponentsContext) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
		this.directedComponentsContext = directedComponentsContext;
	}
	
	@Override
	public ComponentsSubGraphBuilder<C, T> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new ComponentsSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), directedComponentsContext, new ArrayList<>());
		return currentPath;
	}
}
