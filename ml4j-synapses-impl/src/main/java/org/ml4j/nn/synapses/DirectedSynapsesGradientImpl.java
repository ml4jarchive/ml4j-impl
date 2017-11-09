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
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of DirectedSynapsesGradient.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesGradientImpl implements DirectedSynapsesGradient {

  private Matrix axonsGradient;
  private NeuronsActivation output;

  public DirectedSynapsesGradientImpl(NeuronsActivation output, Matrix axonsGradient) {
    this.axonsGradient = axonsGradient;
    this.output = output;
  }

  @Override
  public Matrix getTotalTrainableAxonsGradient() {
    return axonsGradient;
  }
  
  @Override
  public Matrix getAverageTrainableAxonsGradient() {
    return axonsGradient == null ? null : axonsGradient.div(axonsGradient.getColumns());
  }

  @Override
  public NeuronsActivation getOutput() {
    return output;
  }

  @Override
  public String toString() {
    return "DirectedSynapsesGradientImpl [axonsGradient=" + axonsGradient.getRows() + ":"
        + axonsGradient.getColumns() + "]";
  }
}
