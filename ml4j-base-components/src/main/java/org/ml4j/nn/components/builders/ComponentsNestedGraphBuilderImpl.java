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
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.base.BaseNestedGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ComponentsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.ComponentsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public abstract class ComponentsNestedGraphBuilderImpl<P extends ComponentsContainer<Neurons>, C extends AxonsBuilder> extends BaseNestedGraphBuilderImpl<P, C> implements ComponentsGraphBuilder<C>{
	

	public ComponentsNestedGraphBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> withParallelPaths() {
		addAxonsIfApplicable();
		return new ComponentsParallelPathsBuilderImpl<>(directedComponentFactory, this::getBuilder);
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<C> withSkipConnection() {
		return new ComponentsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return getBuilder();
	}
}
