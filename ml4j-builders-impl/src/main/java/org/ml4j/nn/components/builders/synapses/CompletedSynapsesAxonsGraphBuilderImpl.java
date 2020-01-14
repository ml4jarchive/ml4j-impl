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
package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

public class CompletedSynapsesAxonsGraphBuilderImpl<P extends AxonsBuilder<T>, T extends NeuralComponent> extends BaseGraphBuilderImpl<CompletedSynapsesAxonsGraphBuilder<P, T>, T> implements CompletedSynapsesAxonsGraphBuilder<P, T>, SynapsesEnder<P> {

	private Supplier<P> previousSupplier;
	
	public CompletedSynapsesAxonsGraphBuilderImpl(Supplier<P> previousSupplier, NeuralComponentFactory<T> directedComponentFactory,
			BaseGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		super(directedComponentFactory, builderState, directedComponentsContext, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<CompletedSynapsesAxonsGraphBuilder<P, T>, T>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory,() -> this, directedComponentsContext);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<CompletedSynapsesAxonsGraphBuilder<P, T>, T> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
	}

	@Override
	public SynapsesEnder<P> withActivationFunction(
			DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return this;
	}
	
	@Override
	public SynapsesEnder<P> withActivationFunction(
			ActivationFunctionType activationFunctionType) {
		addActivationFunction(activationFunctionType);
		return this;
	}

	@Override
	public P endSynapses() {
		addAxonsIfApplicable();
		this.previousSupplier.get().addAxonsIfApplicable();
		this.previousSupplier.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
		this.previousSupplier.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
		this.previousSupplier.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
		// TODO ML Here we would add the synapses instead of the chain
		T chain = directedComponentFactory.createDirectedComponentChain(this.getComponents());
		previousSupplier.get().addComponent(chain);
		
		return previousSupplier.get();
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<P, T> getBuilder() {
		return this;
	}
}
