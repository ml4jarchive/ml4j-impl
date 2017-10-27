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
import org.ml4j.nn.util.NeuralNetUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The default Sigmoid Activation Function.
 * 
 * @author Michael Lavelle
 *
 */
public class SoftmaxActivationFunction implements DifferentiableActivationFunction {

  private static final Logger LOGGER = LoggerFactory.getLogger(SoftmaxActivationFunction.class);

  @Override
  public NeuronsActivation activate(NeuronsActivation input, NeuronsActivationContext context) {
    LOGGER.debug("Activating through SigmoidActivationFunction");
    if (input.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException(
          "Activations passing through activation function should not include a bias unit"
          + " as this has not yet been implemented");
    }
    Matrix sigmoidOfInputActivationsMatrix = NeuralNetUtils.softmax(input.getActivations());
    return new NeuronsActivation(sigmoidOfInputActivationsMatrix, input.isBiasUnitIncluded(),
        input.getFeatureOrientation());
  }

  @Override
  public NeuronsActivation activationGradient(NeuronsActivation outputActivation,
      NeuronsActivationContext context) {
    if (outputActivation.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException(
          "Activations passing through activation function should not include a bias unit"
          + " as this has not yet been implemented");
    }
    LOGGER.debug("Performing softmax gradient of NeuronsActivation");
    return new NeuronsActivation(
        NeuralNetUtils.softmaxGradient(outputActivation.getActivations()),
        outputActivation.isBiasUnitIncluded(), outputActivation.getFeatureOrientation());

  }
}
