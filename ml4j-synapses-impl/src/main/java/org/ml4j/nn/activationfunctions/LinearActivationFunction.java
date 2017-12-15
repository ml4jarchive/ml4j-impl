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

package org.ml4j.nn.activationfunctions;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The default Linear Activation Function - actually the identify function but the common
 * name when used in neural netnets is linear activation function.
 * 
 * @author Michael Lavelle
 *
 */
public class LinearActivationFunction implements DifferentiableActivationFunction {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(LinearActivationFunction.class);

  @Override
  public DifferentiableActivationFunctionActivation activate(NeuronsActivation input, 
      NeuronsActivationContext context) {
    LOGGER.debug("Activating through LinearActivationFunction");
    
    return new DifferentiableActivationFunctionActivationImpl(this, input, 
        new NeuronsActivation(input.getActivations().dup(),
        input.getFeatureOrientation()));
  }

  @Override
  public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation 
      outputActivation,
      NeuronsActivationContext context) {
    
    LOGGER.debug("Performing linear gradient of NeuronsActivation");
   
    Matrix gradient = context.getMatrixFactory()
        .createOnes(outputActivation.getInput().getActivations().getRows(), 
        outputActivation.getInput().getActivations().getColumns());
   
    return new NeuronsActivation(
        gradient, 
        outputActivation.getInput().getFeatureOrientation());
  }
}
