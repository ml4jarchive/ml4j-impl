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

import org.ml4j.Matrix;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses that not not contain only Axons, but
 * do contain an ActivationFunction.
 * 
 * @author Michael Lavelle
 */
public class ActivationFunctionOnlyDirectedSynapsesImpl
      <L extends Neurons, R extends Neurons> implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(ActivationFunctionOnlyDirectedSynapsesImpl.class);
  
  private L leftNeurons;
  private R rightNeurons;
  private DifferentiableActivationFunction activationFunction;


  /**
   * @param leftNeurons The left neurons
   * @param rightNeurons The right neurons
   * @param activationFunction The activation function.
   */
  public ActivationFunctionOnlyDirectedSynapsesImpl(L leftNeurons, R rightNeurons, 
          DifferentiableActivationFunction activationFunction) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.activationFunction = activationFunction;
  }
  
  @Override
  public DirectedSynapses<L, R> dup() {
    return new ActivationFunctionOnlyDirectedSynapsesImpl<L, R>(leftNeurons, 
        rightNeurons, activationFunction);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesActivation activation,
      NeuronsActivation da, DirectedSynapsesContext context, 
      boolean outerMostSynapses, double arg4) {
    
    LOGGER.debug("Back propagating through synapses activation....");
   
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }
    
    if (activation.getAxonsActivation() != null) {
      throw new IllegalStateException(
          "The synapses activation is not expected to contain an AxonsActivation");
    }
    
    NeuronsActivation activationFunctionInput = activation.getInput();
    
    Matrix dz = null;
    
    if (outerMostSynapses) {
      dz = da.getActivations();
    } else {
      Matrix activationGradient = activationFunction
          .activationGradient(activationFunctionInput, context)
          .getActivations();

      dz = da.getActivations().mul(activationGradient);
    }
  
    if (da.getFeatureCount() != rightNeurons
        .getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Expected feature count to be:"
          + rightNeurons.getNeuronCountExcludingBias() + " but was:"
          + da.getFeatureCount());
    }
    
    // Does not contain output bias unit
    NeuronsActivation dzN = new NeuronsActivation(dz,
        da.getFeatureOrientation());
 
    return new DirectedSynapsesGradientImpl(dzN, 
        null);  
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput input,
      DirectedSynapsesContext synapsesContext) {
  
    NeuronsActivation inputNeuronsActivation = input.getInput();
 
    LOGGER.debug("Forward propagating through DirectedSynapses");
    
    NeuronsActivation activationInput = inputNeuronsActivation;
    
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalStateException("Right neurons with bias not supported");
    }
    
    NeuronsActivation outputNeuronsActivation = 
        activationFunction.activate(activationInput, 
            synapsesContext);
    
    return new DirectedSynapsesActivationImpl(this, 
        inputNeuronsActivation, null, outputNeuronsActivation); 
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
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }

  @Override
  public Axons<?, ?, ?> getAxons() {
    return null;
  }
}
