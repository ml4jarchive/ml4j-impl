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
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public abstract class AxonsGraphBuilderImpl<C extends AxonsBuilder<T>, T extends NeuralComponent<?>>
		extends BaseGraphBuilderImpl<C, T> implements AxonsGraphBuilder<C, T> {

	protected Supplier<C> previousSupplier;
	protected C builder;

	public AxonsGraphBuilderImpl(Supplier<C> previousSupplier, NeuralComponentFactory<T> directedComponentFactory,
			BaseGraphBuilderState builderState, 
			List<T> components) {
		super(directedComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<C, T>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory, previousSupplier);
	}

	@Override
	public AxonsGraphSkipConnectionBuilder<C, T> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(previousSupplier, directedComponentFactory, builderState, new ArrayList<>());
	}
}
