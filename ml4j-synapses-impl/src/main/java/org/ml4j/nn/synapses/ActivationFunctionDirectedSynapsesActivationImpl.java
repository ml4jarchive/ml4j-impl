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

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of DirectedSynapsesActivation for Synapses
 * containing only an ActivationFunction.
 * 
 * @author Michael Lavelle
 */
public class ActivationFunctionDirectedSynapsesActivationImpl 
    extends DirectedSynapsesActivationBase {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(ActivationFunctionDirectedSynapsesActivationImpl.class);

  /**
   * 
   * @param synapses The synapses.
   * @param inputActivation The input activation.
   * @param axonsActivation The axons activation.
   * @param activationFunctionActivation The activation function activation.
   * @param outputActivation The output activation.
   */
  public ActivationFunctionDirectedSynapsesActivationImpl(DirectedSynapses<?, ?> synapses,
      DirectedSynapsesInput inputActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation) {
    super(synapses, inputActivation, null, activationFunctionActivation,
        outputActivation);
  }


  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesGradient da,
      DirectedSynapsesContext context) {

    LOGGER.debug("Back propagating through synapses activation....");

    NeuronsActivation dz = activationFunctionActivation.backPropagate(da, context).getOutput();

    return new DirectedSynapsesGradientImpl(dz, null, 
        inputActivation.getResidualInput() != null ? dz : null);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(CostFunctionGradient da,
      DirectedSynapsesContext context) {

    LOGGER.debug("Back propagating through synapses activation....");

    NeuronsActivation dz = activationFunctionActivation.backPropagate(da, context).getOutput();

    return new DirectedSynapsesGradientImpl(dz, null, 
        inputActivation.getResidualInput() != null ? dz : null);
  }
}
