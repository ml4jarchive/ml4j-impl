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

import org.ml4j.nn.components.builders.Components3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.Base3DGraphBuilderStateImpl;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

public class InitialComponents3DGraphBuilderImpl extends Components3DGraphBuilderImpl<InitialComponents3DGraphBuilder, InitialComponentsGraphBuilder> implements InitialComponents3DGraphBuilder{
	
	private InitialComponentsGraphBuilder nestedBuilder;
	
	public InitialComponents3DGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, Neurons3D currentNeurons) {
		super(directedComponentFactory, new Base3DGraphBuilderStateImpl(currentNeurons), new ArrayList<>());
	}

	@Override
	public InitialComponents3DGraphBuilder get3DBuilder() {
		return this;
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		if (nestedBuilder != null) {
			return nestedBuilder;
		} else {
			nestedBuilder = new InitialComponentsGraphBuilderImpl(directedComponentFactory, builderState.getNon3DBuilderState(), getComponents());
			return nestedBuilder;
		}
	}
}
