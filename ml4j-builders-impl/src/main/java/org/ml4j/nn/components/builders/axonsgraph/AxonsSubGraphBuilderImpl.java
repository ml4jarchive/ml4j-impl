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

import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public class AxonsSubGraphBuilderImpl<C extends ComponentsContainer<Neurons, T>, T extends NeuralComponent<?>>
		extends BaseNestedGraphBuilderImpl<C, AxonsSubGraphBuilder<C, T>, T>
		implements AxonsSubGraphBuilder<C, T>, PathEnder<C, AxonsSubGraphBuilder<C, T>> {

	protected C builder;

	public AxonsSubGraphBuilderImpl(Supplier<C> previousSupplier, NeuralComponentFactory<T> directedComponentFactory,
			BaseGraphBuilderState builderState, 
			List<T> components) {
		super(previousSupplier, directedComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<AxonsSubGraphBuilder<C, T>, T>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory, this::getBuilder);
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<AxonsSubGraphBuilder<C, T>, T> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(() -> this, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public AxonsSubGraphBuilder<C, T> getBuilder() {
		return this;
	}

	@Override
	public AxonsSubGraphBuilder<C, T> withPath() {
		return new AxonsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public PathEnder<C, AxonsSubGraphBuilder<C, T>> endPath() {
		return this;
	}

	@Override
	protected AxonsSubGraphBuilder<C, T> createNewNestedGraphBuilder() {
		return new AxonsSubGraphBuilderImpl<>(parentGraph, directedComponentFactory, initialBuilderState, components);
	}
}
