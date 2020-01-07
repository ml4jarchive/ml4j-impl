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

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.builders.base.Base3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Components3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public abstract class Components3DGraphBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> extends Base3DGraphBuilderImpl<C, D> 
implements Components3DGraphBuilder<C, D> {
	
	public Components3DGraphBuilderImpl(DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
	}

	@Override
	public ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> withParallelPaths() {
		addAxonsIfApplicable();
		return new Components3DParallelPathsBuilderImpl<>(directedComponentFactory, this::get3DBuilder);
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<C, D> withSkipConnection() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(this::get3DBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return get3DBuilder();
	}

	@Override
	public Components3DGraphBuilder<C, D> withComponents(Components3DGraphBuilder<?, ?> builder) {
		addComponents(builder.getComponents());
		return this;
	}
	
	
}
