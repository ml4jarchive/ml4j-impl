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

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of DirectedSynapsesGradient.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesGradientImpl implements DirectedSynapsesGradient {

  private AxonsGradient axonsGradient;
  private NeuronsActivation output;

  public DirectedSynapsesGradientImpl(NeuronsActivation output, AxonsGradient axonsGradient) {
    this.axonsGradient = axonsGradient;
    this.output = output;
  }

  @Override
  public AxonsGradient getTotalTrainableAxonsGradient() {
    return axonsGradient;
  }
  
  @Override
  public AxonsGradient getAverageTrainableAxonsGradient() {
    return axonsGradient == null ? null : 
      new AxonsGradientImpl(axonsGradient.getAxons(), 
          axonsGradient.getGradient()
          .div(axonsGradient.getGradient().getColumns()));
  }

  @Override
  public NeuronsActivation getOutput() {
    return output;
  }

  @Override
  public String toString() {
    return "DirectedSynapsesGradientImpl [axonsGradient=" 
          + axonsGradient.getGradient().getRows() + ":"
        + axonsGradient.getGradient().getColumns() + "]";
  }
}
