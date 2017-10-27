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

package org.ml4j.nn.layers;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.neurons.Neurons;

/**
 * Default implementation of a FullyConnectedFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class FullyConnectedFeedForwardLayerImpl 
    extends FeedForwardLayerBase<FullyConnectedAxons, FullyConnectedFeedForwardLayer>
    implements FullyConnectedFeedForwardLayer {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  public FullyConnectedFeedForwardLayerImpl(FullyConnectedAxons primaryAxons,
      DifferentiableActivationFunction activationFunction) {
    super(primaryAxons, activationFunction);
  }
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param primaryActivationFunction The primary activation function.
   * @param matrixFactory The MatrixFactory to use to initialise the weights
   */
  public FullyConnectedFeedForwardLayerImpl(Neurons inputNeurons, Neurons outputNeurons,
      DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory) {
    super(
        new FullyConnectedAxonsImpl(inputNeurons, outputNeurons,
            matrixFactory),
        primaryActivationFunction);
  }

  @Override
  public FullyConnectedFeedForwardLayer dup() {
    return new FullyConnectedFeedForwardLayerImpl(primaryAxons.dup(), primaryActivationFunction);
  }
}
