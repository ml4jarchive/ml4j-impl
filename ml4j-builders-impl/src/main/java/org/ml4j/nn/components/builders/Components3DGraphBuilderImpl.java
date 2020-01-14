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
import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.base.Base3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.Components3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.skipconnection.Components3DGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class Components3DGraphBuilderImpl<C extends Components3DGraphBuilder<C, D, T>, D extends ComponentsGraphBuilder<D, T>, T extends NeuralComponent> extends Base3DGraphBuilderImpl<C, D, T> 
implements Components3DGraphBuilder<C, D, T> {

	public Components3DGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(directedComponentFactory, builderState, directedComponentsContext, components);
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
	
	@Override
	public C withActivationFunction(ActivationFunctionType activationFunctionType) {
		addActivationFunction(activationFunctionType);
		return get3DBuilder();
	}
	

	@Override
	public C withComponentDefinition(
			Component3Dto3DGraphDefinition componentDefinition) {
		addAxonsIfApplicable();
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder, directedComponentFactory).getComponents());
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setCurrentNeurons(componentDefinition.getOutputNeurons());
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return get3DBuilder();
	}

	@Override
	public D withComponentDefinition(Component3DtoNon3DGraphDefinition componentDefinition) {
		addAxonsIfApplicable();
		InitialComponents3DGraphBuilder<T> builder = new InitialComponents3DGraphBuilderImpl<T>(directedComponentFactory, directedComponentsContext, componentDefinition.getInputNeurons());
		addComponents(componentDefinition.createComponentGraph(builder, directedComponentFactory).getComponents());
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		D nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons().setCurrentNeurons(componentDefinition.getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public C with3DComponent(T component, Neurons3D endNeurons) {
		addAxonsIfApplicable();
		addComponents(Arrays.asList(component));
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setCurrentNeurons(endNeurons);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return get3DBuilder();
	}

	@Override
	public D withNon3DComponent(T component) {
		addAxonsIfApplicable();
		addComponents(Arrays.asList(component));
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		D nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons().setCurrentNeurons(component.getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public D withComponents(ComponentsGraphBuilder<?, T> builder) {
		addAxonsIfApplicable();
		List<T> componentsToAdd = builder.getComponents();
		addComponents(componentsToAdd);
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		D nextBuilder = getBuilder();
		nextBuilder.getComponentsGraphNeurons().setCurrentNeurons(componentsToAdd.get(componentsToAdd.size() - 1).getOutputNeurons());
		return nextBuilder;
	}

	@Override
	public C withComponents(Components3DGraphBuilder<?, ?, T> builder,
			Neurons3D endNeurons) {
		addAxonsIfApplicable();
		List<T> componentsToAdd = builder.getComponents();
		addComponents(componentsToAdd);
		builderState.getComponentsGraphNeurons().setRightNeurons(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		builderState.setConnectionWeights(null);
		return get3DBuilder();
	}
	
	
	
}
