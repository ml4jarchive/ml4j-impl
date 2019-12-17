/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.synapses;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.defaults.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesImpl<L extends Neurons, R extends Neurons> implements DirectedSynapses<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedSynapsesImpl.class);

	private DifferentiableActivationFunction activationFunction;
	private DefaultDirectedComponentBipoleGraph axonsGraph;

	private DirectedComponentFactory directedComponentFactory;
	
	private L leftNeurons;
	private R rightNeurons;

	/**
	 * Create a new implementation of DirectedSynapses.
	 * 
	 * @param directedComponentFactory
	 * @param LeftNeurons
	 * @param rightNeurons
	 * @param axonsGraph
	 * @param activationFunction
	 */
	protected DirectedSynapsesImpl(DirectedComponentFactory directedComponentFactory, L leftNeurons, R rightNeurons,
			DefaultDirectedComponentBipoleGraph axonsGraph,
			DifferentiableActivationFunction activationFunction) {
		super();
		this.activationFunction = activationFunction;
		this.axonsGraph = axonsGraph;
		this.directedComponentFactory = directedComponentFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	/**
	 * Create a new implementation of DirectedSynapses.
	 * 
	 * @param directedComponentFactory A factory implementation to create directed
	 *                                 components
	 * @param primaryAxons             The primary Axons within these synapses
	 * @param activationFunction       The activation function within these synapses
	 */
	public DirectedSynapsesImpl(DirectedComponentFactory directedComponentFactory,
			Axons<? extends L, ? extends R, ?> primaryAxons, DifferentiableActivationFunction activationFunction) {
		this(directedComponentFactory, primaryAxons.getLeftNeurons(), primaryAxons.getRightNeurons(), createGraph(directedComponentFactory, primaryAxons),
				activationFunction);
		this.directedComponentFactory = directedComponentFactory;
	}

	private static DefaultDirectedComponentBipoleGraph createGraph(
			DirectedComponentFactory directedComponentFactory, Axons<?, ?, ?> primaryAxons) {
		List<DefaultChainableDirectedComponent<?,  ?>> components = Arrays
				.asList(directedComponentFactory.createDirectedAxonsComponent(primaryAxons));
		DefaultDirectedComponentChain chain = new DefaultDirectedComponentChainImpl(
				components);
		List<DefaultDirectedComponentChain> chainsList = new ArrayList<>();
		chainsList.add(chain);
		DefaultDirectedComponentChainBatch<?, ?> batch = new DefaultDirectedComponentChainBatchImpl<>(chainsList);
		return new DefaultDirectedComponentChainBipoleGraphImpl<>(directedComponentFactory, batch,
				PathCombinationStrategy.ADDITION);
	}

	/**
	 * @return The Axons graph within these DirectedSynapses.
	 */
	public DefaultDirectedComponentBipoleGraph getAxonsGraph() {
		return axonsGraph;
	}

	@Override
	public DirectedSynapses<L, R> dup() {
		return new DirectedSynapsesImpl<>(directedComponentFactory, leftNeurons, rightNeurons, axonsGraph.dup(),
				activationFunction);
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public DirectedSynapsesActivation forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext directedComponentsContext) {

		LOGGER.debug("Forward propagating through DirectedSynapses");

		NeuronsActivation inputNeuronsActivation = input;

		DefaultDirectedComponentBipoleGraphActivation axonsActivationGraph = axonsGraph
				.forwardPropagate(inputNeuronsActivation, directedComponentsContext);

		NeuronsActivation totalAxonsOutputActivation = axonsActivationGraph.getOutput();

		DifferentiableActivationFunctionComponent actComp = directedComponentFactory
				.createDifferentiableActivationFunctionComponent(activationFunction);

		DifferentiableActivationFunctionActivation actAct = actComp.forwardPropagate(totalAxonsOutputActivation,
				new NeuronsActivationContext() {

					/**
					 * 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public MatrixFactory getMatrixFactory() {
						return directedComponentsContext.getMatrixFactory();
					}

				});

		NeuronsActivation outputNeuronsActivation = actAct.getOutput();

		return new DirectedSynapsesActivationImpl(this, input, axonsActivationGraph, actAct, outputNeuronsActivation,
				directedComponentsContext);

	}

	@Override
	public L getLeftNeurons() {

		return leftNeurons;
	}

	@Override
	public R getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext,
			int componentIndex) {
		return directedComponentsContext;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		
		List<DefaultChainableDirectedComponent<?, ?>> components = 
				new ArrayList<>();
		components.addAll(axonsGraph.decompose().stream().map(c -> adaptComponent(c)).collect(Collectors.toList()));
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(activationFunction));
		return components;
	}
	
	 protected <C> DefaultChainableDirectedComponentAdapter<?> adaptComponent(ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> c) {
		  return new DefaultChainableDirectedComponentAdapter<>(c);
	  }

}
