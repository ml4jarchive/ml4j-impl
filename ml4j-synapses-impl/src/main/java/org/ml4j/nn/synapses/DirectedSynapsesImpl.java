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
import java.util.Objects;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
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
	private DifferentiableActivationFunctionComponent activationFunctionComponent;

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
		this.activationFunctionComponent = directedComponentFactory
				.createDifferentiableActivationFunctionComponent(activationFunction);
		this.axonsGraph = axonsGraph;
		this.directedComponentFactory = directedComponentFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		Objects.requireNonNull(axonsGraph, "axonsGraph");

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
		DefaultDirectedComponentChain chain = directedComponentFactory.createDirectedComponentChain(
				components);
		List<DefaultDirectedComponentChain> chainsList = new ArrayList<>();
		chainsList.add(chain);
		DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> batch = directedComponentFactory.createDirectedComponentChainBatch(chainsList);
		return directedComponentFactory.createDirectedComponentBipoleGraph(batch,
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

		DifferentiableActivationFunctionActivation actAct = activationFunctionComponent.forwardPropagate(totalAxonsOutputActivation,
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
		components.addAll(axonsGraph.decompose().stream().collect(Collectors.toList()));
		components.addAll(activationFunctionComponent.decompose());
		return components;
	}

	  @Override
	  public DirectedComponentType getComponentType() {
		  return DirectedComponentType.SYNAPSES;
	  }

}
