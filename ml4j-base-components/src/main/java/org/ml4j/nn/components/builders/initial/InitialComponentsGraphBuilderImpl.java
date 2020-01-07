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
package org.ml4j.nn.components.builders.initial;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderStateImpl;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public class InitialComponentsGraphBuilderImpl extends ComponentsGraphBuilderImpl<InitialComponentsGraphBuilder> implements InitialComponentsGraphBuilder {

	public InitialComponentsGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState, List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
	}
	
	public InitialComponentsGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, Neurons initialNeurons) {
		super(directedComponentFactory, new BaseGraphBuilderStateImpl(initialNeurons), new ArrayList<>());
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		return this;
	}
}
