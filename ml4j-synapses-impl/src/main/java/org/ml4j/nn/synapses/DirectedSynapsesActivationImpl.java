/*
 * Copyright 2017 the original author or authors.
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

package org.ml4j.nn.synapses;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesActivationImpl extends DirectedSynapsesActivationBase {

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedSynapsesActivationImpl.class);

	/**
	 * 
	 * @param synapses                     The synapses.
	 * @param inputActivation              The input activation.
	 * @param axonsActivation              The axons activation.
	 * @param activationFunctionActivation The activation function activation.
	 * @param outputActivation             The output activation.
	 */
	public DirectedSynapsesActivationImpl(DirectedSynapses<?, ?> synapses, NeuronsActivation inputActivation,
			DefaultDirectedComponentBipoleGraphActivation axonsActivation,
			DifferentiableActivationFunctionComponentActivation activationFunctionActivation, NeuronsActivation outputActivation,
			DirectedComponentsContext synapsesContext) {
		super(synapses, inputActivation, axonsActivation, activationFunctionActivation, outputActivation,
				synapsesContext);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> da) {

		LOGGER.debug("Back propagating through synapses activation....");

		validateAxonsAndAxonsActivation();

		DirectedComponentGradient<NeuronsActivation> dz = activationFunctionActivation.backPropagate(da);

		return axonsActivationGraph.backPropagate(dz);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient da) {

		LOGGER.debug("Back propagating through synapses activation....");

		validateAxonsAndAxonsActivation();

		DirectedComponentGradient<NeuronsActivation> dz = activationFunctionActivation.backPropagate(da);

		return axonsActivationGraph.backPropagate(dz);
	}

	private void validateAxonsAndAxonsActivation() {

		if (synapses.getRightNeurons().hasBiasUnit()) {
			throw new IllegalStateException("Backpropagation through axons with a rhs bias unit not supported");
		}
	}
}
