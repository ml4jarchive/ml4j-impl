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

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderStateImpl;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public class InitialComponentsGraphBuilderImpl<T extends NeuralComponent> extends ComponentsGraphBuilderImpl<InitialComponentsGraphBuilder<T>, T> implements InitialComponentsGraphBuilder<T> {

	public InitialComponentsGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory, BaseGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,List<T> components) {
		super(directedComponentFactory, builderState, directedComponentsContext, components);
	}
	
	public InitialComponentsGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory, DirectedComponentsContext directedComponentsContext,Neurons initialNeurons) {
		super(directedComponentFactory, new BaseGraphBuilderStateImpl(initialNeurons), directedComponentsContext, new ArrayList<>());
	}

	@Override
	public InitialComponentsGraphBuilder<T> getBuilder() {
		return this;
	}
}
