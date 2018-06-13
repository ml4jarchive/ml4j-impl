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
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedDipoleGraphImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of DirectedSynapses containing an
 * ActivationFunction only.
 * 
 * @author Michael Lavelle
 */
public class ActivationFunctionDirectedSynapsesImpl<L extends Neurons, R extends Neurons> 
    implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(ActivationFunctionDirectedSynapsesImpl.class);
  
  private DifferentiableActivationFunction activationFunction;
  private L leftNeurons;
  private R rightNeurons;
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param activationFunction The activation function within these synapses
   */
  public ActivationFunctionDirectedSynapsesImpl(L leftNeurons, R rightNeurons,
      DifferentiableActivationFunction activationFunction) {
    super();
    this.activationFunction = activationFunction;
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
  }
 

  @Override
  public Axons<? extends L, ? extends R, ?> getPrimaryAxons() {
    return null;
  }
  
  /**
   * @return The Axons graph within these DirectedSynapses.
   */
  public DirectedDipoleGraph<Axons<?, ?, ?>> getAxonsGraph() {
    return new DirectedDipoleGraphImpl<Axons<?, ?, ?>>();
  }

  @Override
  public DirectedSynapses<L, R> dup() {
    return new ActivationFunctionDirectedSynapsesImpl<L, R>(leftNeurons, 
        rightNeurons, activationFunction);
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }


  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput input,
      DirectedSynapsesContext synapsesContext) {

    LOGGER.debug("Forward propagating through ActivationFunctionDirectedSynapses");


    Matrix combinedInput = input.getInput().getActivations();
    if (input.getResidualInput() != null) {
      combinedInput = combinedInput.add(input.getResidualInput().getActivations());
    }
    
    DifferentiableActivationFunctionActivation activationFunctionActivation =
        activationFunction.activate(new NeuronsActivation(combinedInput, 
            input.getInput().getFeatureOrientation()), synapsesContext);

    NeuronsActivation outputNeuronsActivation = activationFunctionActivation.getOutput();

    return new ActivationFunctionDirectedSynapsesActivationImpl(this, input,
        activationFunctionActivation, outputNeuronsActivation);

  }
  
  protected NeuronsActivation getInputNeuronsActivationForPathIndex(
      DirectedSynapsesInput synapsesInput, int pathIndex) {
    if (pathIndex != 0) {
      throw new IllegalArgumentException("Path index:" + pathIndex + " not valid for "
          + "DirectedSynapsesImpl - custom classes can override this behaviour");
    }
    return synapsesInput.getInput();
  }

  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }
}
