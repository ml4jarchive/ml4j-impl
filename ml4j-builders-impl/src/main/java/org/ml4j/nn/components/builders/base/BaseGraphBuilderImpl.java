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
package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.AxonsContextAwareNeuralComponent;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.AxonsPermitted;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilder;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilderImpl;
import org.ml4j.nn.components.builders.synapses.SynapsesPermitted;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public abstract class BaseGraphBuilderImpl<C extends AxonsBuilder<T>, T extends NeuralComponent>
		implements AxonsPermitted<C>, SynapsesPermitted<C, T>, AxonsBuilder<T> {

	protected NeuralComponentFactory<T> directedComponentFactory;

	protected BaseGraphBuilderState builderState;

	protected BaseGraphBuilderState initialBuilderState;

	protected DirectedComponentsContext directedComponentsContext;

	private List<T> chains;
	private List<Neurons> endNeurons;

	protected List<T> components;

	public BaseGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory, BaseGraphBuilderState builderState,
			DirectedComponentsContext directedComponentsContext, List<T> components) {
		this.builderState = builderState;
		this.components = components;
		this.directedComponentFactory = directedComponentFactory;
		this.initialBuilderState = new BaseGraphBuilderStateImpl(
				builderState.getComponentsGraphNeurons().getCurrentNeurons());
		this.chains = new ArrayList<>();
		this.endNeurons = new ArrayList<>();
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public List<T> getComponents() {
		return components;
	}

	@Override
	public ComponentsContainer<Neurons, T> getAxonsBuilder() {
		return this;
	}

	public BaseGraphBuilderState getBuilderState() {
		return builderState;
	}

	@Override
	public ComponentsGraphNeurons<Neurons> getComponentsGraphNeurons() {
		return builderState.getComponentsGraphNeurons();
	}

	@Override
	public Matrix getConnectionWeights() {
		return builderState.getConnectionWeights();
	}

	public AxonsBuilder<T> withConnectionWeights(Matrix connectionWeights) {
		builderState.setConnectionWeights(connectionWeights);
		return this;
	}

	public abstract C getBuilder();

	public void addAxonsIfApplicable() {
		if ((builderState.getFullyConnectedAxonsBuilder() != null)
				&& builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Neurons leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons(
						builderState.getComponentsGraphNeurons().getCurrentNeurons().getNeuronCountExcludingBias(),
						true);
			}

			T axonsComponent = directedComponentFactory.createFullyConnectedAxonsComponent(leftNeurons,
					builderState.getComponentsGraphNeurons().getRightNeurons(), builderState.getConnectionWeights(),
					builderState.getBiases());

			if (builderState.getFullyConnectedAxonsBuilder().getAxonsContextConfigurer() != null) {
				// TODO
				if (axonsComponent instanceof AxonsContextAwareNeuralComponent) {
					AxonsContext axonsContext = ((AxonsContextAwareNeuralComponent) axonsComponent)
							.getContext(directedComponentsContext, 0);
					builderState.getFullyConnectedAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
				}
			}

			components.add(axonsComponent);

			builderState.setFullyConnectedAxonsBuilder(null);
			builderState.getComponentsGraphNeurons()
					.setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
		}
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withFullyConnectedAxons() {
		addAxonsIfApplicable();
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		UncompletedFullyConnectedAxonsBuilder<C> axonsBuilder = new UncompletedFullyConnectedAxonsBuilderImpl<>(
				this::getBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setFullyConnectedAxonsBuilder(axonsBuilder);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public SynapsesAxonsGraphBuilder<C, T> withSynapses() {
		addAxonsIfApplicable();
		SynapsesAxonsGraphBuilder<C, T> synapsesBuilder = new SynapsesAxonsGraphBuilderImpl<>(this::getBuilder,
				directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
		builderState.setSynapsesBuilder(synapsesBuilder);
		return synapsesBuilder;
	}

	protected void addActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(
				this.builderState.getComponentsGraphNeurons().getCurrentNeurons(), activationFunction));
	}

	protected void addActivationFunction(ActivationFunctionType activationFunctionType) {
		addAxonsIfApplicable();
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(
				this.builderState.getComponentsGraphNeurons().getCurrentNeurons(), activationFunctionType));
	}

	public T getComponentChain() {
		addAxonsIfApplicable();
		return directedComponentFactory.createDirectedComponentChain(components);
	}

	public AxonsBuilder<T> withBiasUnit() {
		builderState.getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public void addComponents(List<T> components) {
		this.components.addAll(components);
	}

	@Override
	public void addComponent(T component) {
		this.components.add(component);
	}

	public List<T> getChains() {
		return chains;
	}

	public List<Neurons> getEndNeurons() {
		return endNeurons;
	}

}
