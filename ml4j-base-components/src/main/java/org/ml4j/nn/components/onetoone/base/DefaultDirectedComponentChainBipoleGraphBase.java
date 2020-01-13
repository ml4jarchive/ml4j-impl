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
package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.neurons.Neurons;

/**
 * Default base class for implementations of DefaultDirectedComponentBipoleGraph
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentChainBipoleGraph {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected DefaultDirectedComponentChainBatch parallelComponentChainsBatch;

	protected Neurons inputNeurons;
	protected Neurons outputNeurons;
	
	/**
	 * @param inputNeurons The input neurons of this graph.
	 * @param outputNeurons The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this graph, connecting
	 * the input neurons to the output neurons.
	 */
	public DefaultDirectedComponentChainBipoleGraphBase(Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch) {
		this.parallelComponentChainsBatch = parallelComponentChainsBatch;
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
	}
	
	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return context;
	}
	@Override
	public NeuralComponentType<DefaultDirectedComponentChainBipoleGraph> getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentType.getBaseType(NeuralComponentBaseType.COMPONENT_CHAIN_BIPOLE_GRAPH), DefaultDirectedComponentChainBipoleGraph.class.getName());
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

	@Override
	public DefaultDirectedComponentChainBatch getEdges() {
		return parallelComponentChainsBatch;
	}
	
	
}
