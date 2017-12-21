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

package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;

/**
 * Encapsulates the artifacts produced when pushing NeuronsActivations
 * through an Axons instance.
 * 
 * @author Michael Lavelle
 */
public class AxonsActivationImpl implements AxonsActivation {

  private Axons<?, ?, ?> axons;
  private Matrix inputDropoutMask;
  private NeuronsActivation outputActivations;
  private NeuronsActivationWithPossibleBiasUnit postDropoutInput;
  
  /**
   * @param inputDropoutMask Any input dropout mask
   * @param postDropoutInput The post dropout input
   * @param outputActivations The output.
   */
  public AxonsActivationImpl(Axons<?, ?, ?> axons, Matrix inputDropoutMask, 
      NeuronsActivationWithPossibleBiasUnit postDropoutInput, NeuronsActivation outputActivations) {
    this.outputActivations = outputActivations;
    this.inputDropoutMask = inputDropoutMask;
    this.postDropoutInput = postDropoutInput;
    this.axons = axons;
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivations;
  }

  @Override
  public Matrix getInputDropoutMask() {
    return inputDropoutMask;
  }

  @Override
  public NeuronsActivationWithPossibleBiasUnit getPostDropoutInputWithPossibleBias() {
    return postDropoutInput;
  }

  public Axons<?, ?, ?> getAxons() {
    return axons;
  }
}
