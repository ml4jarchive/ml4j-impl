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

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public abstract class DirectedSynapsesActivationBase implements DirectedSynapsesActivation {

	protected NeuronsActivation inputActivation;
	protected DefaultDirectedComponentBipoleGraphActivation axonsActivationGraph;
	protected DifferentiableActivationFunctionComponentActivation activationFunctionActivation;
	protected NeuronsActivation outputActivation;
	protected DirectedSynapses<?, ?> synapses;
	protected DirectedComponentsContext synapsesContext;

	/**
	 * Construct a new default DirectedSynapsesActivation
	 * 
	 * @param synapses             The DirectedSynapses
	 * @param inputActivation      The input NeuronsActivation of the
	 *                             DirectedSynapses following a forward propagation
	 * @param axonsActivationGraph The axons NeuronsActivation graph of the
	 *                             DirectedSynapses following a forward propagation
	 * @param outputActivation     The output NeuronsActivation of the
	 *                             DirectedSynapses following a forward propagation.
	 */
	public DirectedSynapsesActivationBase(DirectedSynapses<?, ?> synapses, NeuronsActivation inputActivation,
			DefaultDirectedComponentBipoleGraphActivation axonsActivationGraph,
			DifferentiableActivationFunctionComponentActivation activationFunctionActivation,
			NeuronsActivation outputActivation, DirectedComponentsContext synapsesContext) {
		this.inputActivation = inputActivation;
		this.outputActivation = outputActivation;
		this.synapses = synapses;
		this.axonsActivationGraph = axonsActivationGraph;
		this.activationFunctionActivation = activationFunctionActivation;
		this.synapsesContext = synapsesContext;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(axonsActivationGraph, activationFunctionActivation);
	}

	@Override
	public NeuronsActivation getOutput() {
		return outputActivation;
	}

	@Override
	public DirectedSynapses<?, ?> getSynapses() {
		return synapses;
	}

}
