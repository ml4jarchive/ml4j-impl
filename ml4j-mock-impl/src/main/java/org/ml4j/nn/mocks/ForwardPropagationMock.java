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

package org.ml4j.nn.mocks;

import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Mock implementation of ForwardPropagation.
 * 
 * @author Michael Lavelle
 */
public class ForwardPropagationMock implements ForwardPropagation {
  
  private NeuronsActivation outputActivations;
  
  /**
   * Create a new mock ForwardPropagation instance from the output activations at the
   * right hand side of a DirectedNeuralNetwork after a forward propagation.
   * 
   * @param outputActivations The output activations at the
   *        right hand side of a DirectedNeuralNetwork after a forward propagation.
   */
  public ForwardPropagationMock(NeuronsActivation outputActivations) {
    super();
    this.outputActivations = outputActivations;
  }

  @Override
  public NeuronsActivation getOutputs() {
    return outputActivations;
  }
}
