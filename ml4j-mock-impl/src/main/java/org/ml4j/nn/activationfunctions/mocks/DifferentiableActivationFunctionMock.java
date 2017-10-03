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

package org.ml4j.nn.activationfunctions.mocks;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of DifferentiableActivationFunction.
 * 
 * @author Michael Lavelle
 *
 */
public class DifferentiableActivationFunctionMock implements DifferentiableActivationFunction {

  private static final Logger LOGGER = LoggerFactory.getLogger(
      DifferentiableActivationFunctionMock.class);
  
  @Override
  public NeuronsActivation activate(NeuronsActivation input, NeuronsActivationContext context) {
    LOGGER.debug("Mock activating through DifferentiableActivationFunctionMock");
    return input;
  }

  @Override
  public NeuronsActivation activationGradient(NeuronsActivation outputActivation,
      NeuronsActivationContext context) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
