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

import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;

public class AxonsParallelPathsBuilderImpl<C extends AxonsBuilder> implements ParallelPathsBuilder<AxonsSubGraphBuilder<C>> {

	private DirectedComponentFactory directedComponentFactory;
	private Supplier<C> previousSupplier;
	private AxonsSubGraphBuilder<C> currentPath;
	
	public AxonsParallelPathsBuilderImpl(DirectedComponentFactory directedComponentFactory, Supplier<C> previousSupplier) {
		this.directedComponentFactory = directedComponentFactory;
		this.previousSupplier = previousSupplier;
	}
	
	@Override
	public AxonsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new AxonsSubGraphBuilderImpl<>(previousSupplier, directedComponentFactory, previousSupplier.get().getBuilderState(), new ArrayList<>());
		return currentPath;
	}
}
