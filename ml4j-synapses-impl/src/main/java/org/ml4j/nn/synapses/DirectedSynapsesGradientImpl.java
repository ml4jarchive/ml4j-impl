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

import java.util.ArrayList;
import java.util.List;

/**
 * Default implementation of DirectedSynapsesGradient.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesGradientImpl implements DirectedSynapsesGradient {

  private List<AxonsGradient> axonsGradients;
  private NeuronsActivation output;

  public DirectedSynapsesGradientImpl(NeuronsActivation output, 
      List<AxonsGradient> axonsGradients) {
    this.axonsGradients = axonsGradients;
    this.output = output;
  }

  @Override
  public List<AxonsGradient> getTotalTrainableAxonsGradients() {
    return axonsGradients;
  }
  
  @Override
  public List<AxonsGradient> getAverageTrainableAxonsGradients() {
    
    List<AxonsGradient> averageGradients = new ArrayList<>();
    for (AxonsGradient axonsGradient : axonsGradients) {
      averageGradients.add(new AxonsGradientImpl(axonsGradient.getAxons(), 
          axonsGradient.getGradient().div(axonsGradient.getGradient().getColumns())));
    }
    return averageGradients;
  }

  @Override
  public NeuronsActivation getOutput() {
    return output;
  }

  @Override
  public String toString() {
    return "DirectedSynapsesGradientImpl";
  }
}
