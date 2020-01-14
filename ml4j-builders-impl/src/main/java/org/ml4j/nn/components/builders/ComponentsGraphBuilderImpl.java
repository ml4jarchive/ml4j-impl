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

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public abstract class ComponentsGraphBuilderImpl<C extends AxonsBuilder<T>, T extends NeuralComponent> extends BaseGraphBuilderImpl<C, T> implements ComponentsGraphBuilder<C, T>{
	
	
	public ComponentsGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory,
			BaseGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
			super(directedComponentFactory, builderState, directedComponentsContext, components);
	}

	@Override
	public ParallelPathsBuilder<ComponentsSubGraphBuilder<C, T>> withParallelPaths() {
		addAxonsIfApplicable();
		return new ComponentsParallelPathsBuilderImpl<>(directedComponentFactory, this::getBuilder, directedComponentsContext);
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<C, T> withSkipConnection() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return getBuilder();
	}
	
	@Override
	public C withActivationFunction(ActivationFunctionType activationFunctionType) {
		addActivationFunction(activationFunctionType);
		return getBuilder();
	}
}
