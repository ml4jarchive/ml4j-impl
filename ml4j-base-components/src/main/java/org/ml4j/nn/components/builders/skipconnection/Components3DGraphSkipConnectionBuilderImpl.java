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
package org.ml4j.nn.components.builders.skipconnection;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsNested3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public class Components3DGraphSkipConnectionBuilderImpl<P extends Components3DGraphBuilder<P, Q>, Q extends ComponentsGraphBuilder<Q>>
		extends ComponentsNested3DGraphBuilderImpl<P, Components3DGraphSkipConnectionBuilder<P, Q>, ComponentsGraphSkipConnectionBuilder<Q>>
		implements Components3DGraphSkipConnectionBuilder<P, Q>, PathEnder<P, Components3DGraphSkipConnectionBuilder<P, Q>>{

	private ComponentsGraphSkipConnectionBuilder<Q> builder;
	
	public Components3DGraphSkipConnectionBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(parentGraph, directedComponentFactory, builderState, components);
	}
	

	@Override
	public P endSkipConnection() {
		completeNestedGraph(true);
		completeNestedGraphs(PathCombinationStrategy.ADDITION);
		return parent3DGraph.get();
	}

	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsGraphSkipConnectionBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {
			builder =  new ComponentsGraphSkipConnectionBuilderImpl<>(() -> parent3DGraph.get().getBuilder(), directedComponentFactory,
					builderState.getNon3DBuilderState(), getComponents());
			return builder;
		}
	}


	@Override
	protected Components3DGraphSkipConnectionBuilder<P, Q> createNewNestedGraphBuilder() {
		return new Components3DGraphSkipConnectionBuilderImpl<>(parent3DGraph, directedComponentFactory, initialBuilderState, new ArrayList<>());
	}
	
	@Override
	public Components3DGraphSkipConnectionBuilder<P, Q> withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}


	@Override
	public Components3DGraphBuilder<Components3DGraphSkipConnectionBuilder<P, Q>, ComponentsGraphSkipConnectionBuilder<Q>> withComponents(
			Components3DGraphBuilder<?, ?> builder) {
		addComponents(builder.getComponents());
		return this;
	}

	
}
