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

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesImpl implements DirectedSynapses<Axons<?, ?, ?>> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesImpl.class);
  
  private Axons<?, ?, ?> axons;
  private DifferentiableActivationFunction activationFunction;
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param axons The Axons within these synapses
   * @param activationFunction The activation function within these synapses
   */
  public DirectedSynapsesImpl(Axons<?, ?, ?> axons,
      DifferentiableActivationFunction activationFunction) {
    super();
    this.axons = axons;
    this.activationFunction = activationFunction;
  }

  @Override
  public Axons<?, ?, ?> getAxons() {
    return axons;
  }

  @Override
  public DirectedSynapses<Axons<?, ?, ?>> dup() {
    return new DirectedSynapsesImpl(axons.dup(), activationFunction);
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedSynapsesContext synapsesContext) {
   
    if (!inputNeuronsActivation.isBiasUnitIncluded() && axons.getLeftNeurons().hasBiasUnit()) {
      inputNeuronsActivation = inputNeuronsActivation.withBiasUnit(true, synapsesContext);
    }
   
    LOGGER.debug("Forward propagating through DirectedSynapses");
    AxonsActivation axonsActivation = 
        axons.pushLeftToRight(inputNeuronsActivation, null, 
            synapsesContext.createAxonsContext());
    
    NeuronsActivation axonsOutputActivation = axonsActivation.getOutput();
    
    NeuronsActivation activationFunctionOutputActivation = 
        activationFunction.activate(axonsOutputActivation, synapsesContext);
    
    return new DirectedSynapsesActivationImpl(this, 
        inputNeuronsActivation, axonsActivation, activationFunctionOutputActivation);
  
  }
}
