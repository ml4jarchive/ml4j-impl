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

import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of UndirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class UndirectedSynapsesActivationImpl implements UndirectedSynapsesActivation {

	private NeuronsActivation inputActivation;
	private AxonsActivation axonsActivation;
	private NeuronsActivation outputActivation;
	private UndirectedSynapses<?, ?> synapses;

	/**
	 * Construct a new default DirectedSynapsesActivation
	 * 
	 * @param synapses         The DirectedSynapses
	 * @param inputActivation  The input NeuronsActivation of the DirectedSynapses
	 *                         following a forward propagation
	 * @param axonsActivation  The axons NeuronsActivation of the DirectedSynapses
	 *                         following a forward propagation
	 * @param outputActivation The output NeuronsActivation of the DirectedSynapses
	 *                         following a forward propagation.
	 */
	public UndirectedSynapsesActivationImpl(UndirectedSynapses<?, ?> synapses, NeuronsActivation inputActivation,
			AxonsActivation axonsActivation, NeuronsActivation outputActivation) {
		this.inputActivation = inputActivation;
		this.outputActivation = outputActivation;
		this.synapses = synapses;
		this.axonsActivation = axonsActivation;
	}

	@Override
	public NeuronsActivation getOutput() {
		return outputActivation;
	}

	@Override
	public UndirectedSynapses<?, ?> getSynapses() {
		return synapses;
	}

	@Override
	public AxonsActivation getAxonsActivation() {
		return axonsActivation;
	}

	@Override
	public NeuronsActivation getInput() {
		return inputActivation;
	}
}
