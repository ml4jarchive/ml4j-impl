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

package org.ml4j.layers.mocks;

import org.ml4j.axons.mocks.AxonsMock;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * A minimal mock skeleton FeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class FeedForwardLayerMock implements FeedForwardLayer<FullyConnectedAxons, 
    FeedForwardLayerMock> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private FullyConnectedAxons primaryAxons;
  
  public FeedForwardLayerMock(Neurons inputNeurons, Neurons outputNeurons) {
      this(new AxonsMock(inputNeurons, outputNeurons));
  }
  
  protected FeedForwardLayerMock(FullyConnectedAxons primaryAxons) {
    this.primaryAxons = primaryAxons;
  }

  @Override
  public FeedForwardLayerMock dup() {
    return new FeedForwardLayerMock(primaryAxons);
  }

  @Override
  public int getInputNeuronCount() {
    return primaryAxons.getLeftNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public int getOutputNeuronCount() {
    return primaryAxons.getRightNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public FullyConnectedAxons getPrimaryAxons() {
    return primaryAxons;
  }

  @Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outpuNeuronIndex,
      DirectedLayerContext directedLayerContext) {
    return new NeuronsActivation(directedLayerContext.getMatrixFactory()
        .createZeros(1, getInputNeuronCount()), false, 
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }
}
