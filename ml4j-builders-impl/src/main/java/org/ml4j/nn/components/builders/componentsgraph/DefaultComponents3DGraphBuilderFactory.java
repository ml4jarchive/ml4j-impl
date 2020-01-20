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
package org.ml4j.nn.components.builders.componentsgraph;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

public class DefaultComponents3DGraphBuilderFactory<T extends NeuralComponent>
		implements Components3DGraphBuilderFactory<T> {

	private NeuralComponentFactory<T> directedComponentFactory;

	public DefaultComponents3DGraphBuilderFactory(NeuralComponentFactory<T> directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public InitialComponents3DGraphBuilder<T> createInitialComponents3DGraphBuilder(Neurons3D initialNeurons,
			DirectedComponentsContext directedComponentsContext) {
		return new InitialComponents3DGraphBuilderImpl<>(directedComponentFactory, directedComponentsContext,
				initialNeurons);
	}

}
