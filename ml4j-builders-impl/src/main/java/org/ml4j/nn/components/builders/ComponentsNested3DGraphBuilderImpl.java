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
package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.base.BaseNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Components3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class ComponentsNested3DGraphBuilderImpl<P extends ComponentsContainer<Neurons3D, T>, C extends Components3DGraphBuilder<C, D, T>, D extends ComponentsGraphBuilder<D, T>, T extends NeuralComponent> extends BaseNested3DGraphBuilderImpl<P, C, D, T> 
implements Components3DGraphBuilder<C, D, T> {
	
	public ComponentsNested3DGraphBuilderImpl(Supplier<P> parent3DGraph, NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(parent3DGraph, directedComponentFactory, builderState, directedComponentsContext, components);
		
	}

	@Override
	public ParallelPathsBuilder<Components3DSubGraphBuilder<C, D, T>> withParallelPaths() {
		addAxonsIfApplicable();
		return new Components3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder, directedComponentsContext);
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<C, D, T> withSkipConnection() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return get3DBuilder();
	}
}
