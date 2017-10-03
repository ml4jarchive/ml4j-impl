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

package org.ml4j.nn.synapses.mocks;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesMock implements DirectedSynapses<FullyConnectedAxons> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesMock.class);
  
  private FullyConnectedAxons axons;
  private DifferentiableActivationFunction activationFunction;
  
  /**
   * Create a new mock implementation of DirectedSynapses.
   * 
   * @param axons The Axons within these synapses
   * @param activationFunction The activation function within these synapses
   */
  public DirectedSynapsesMock(FullyConnectedAxons axons,
      DifferentiableActivationFunction activationFunction) {
    super();
    this.axons = axons;
    this.activationFunction = activationFunction;
  }

  @Override
  public FullyConnectedAxons getAxons() {
    return axons;
  }

  @Override
  public DirectedSynapses<FullyConnectedAxons> dup() {
    return new DirectedSynapsesMock(axons.dup(), activationFunction);
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedSynapsesContext synapsesContext) {
   
    LOGGER.debug("Forward propagating through DirectedSynapses");
    NeuronsActivation axonsOutputActivation = 
        axons.pushLeftToRight(inputNeuronsActivation, 
            synapsesContext.createAxonsContext());
    
    NeuronsActivation activationFunctionOutputActivation = 
        activationFunction.activate(axonsOutputActivation, synapsesContext);
    
    return new DirectedSynapsesActivationMock(activationFunctionOutputActivation);
  
  }

}
