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
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.MaxPoolingAxonsImpl;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of a MaxPoolingFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class MaxPoolingFeedForwardLayerImpl 
    extends FeedForwardLayerBase<MaxPoolingAxons, MaxPoolingFeedForwardLayer>
    implements MaxPoolingFeedForwardLayer {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  public MaxPoolingFeedForwardLayerImpl(MaxPoolingAxons primaryAxons) {
    super(primaryAxons, new LinearActivationFunction());
  }
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights
   */
  public MaxPoolingFeedForwardLayerImpl(Neurons3D inputNeurons, 
      Neurons3D outputNeurons, MatrixFactory matrixFactory) {
    super(
        new MaxPoolingAxonsImpl(inputNeurons, outputNeurons,
            matrixFactory, false),
        new LinearActivationFunction());
  }
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights
   * @param scaleOutputs Whether to scale outputs up by a factor equal to the reduction factor due
   *        to only outputting max-elements.
   */
  public MaxPoolingFeedForwardLayerImpl(Neurons3D inputNeurons, 
      Neurons3D outputNeurons, MatrixFactory matrixFactory, boolean scaleOutputs) {
    super(
        new MaxPoolingAxonsImpl(inputNeurons, outputNeurons,
            matrixFactory, scaleOutputs),
        new LinearActivationFunction());
  }

  @Override
  public MaxPoolingFeedForwardLayer dup() {
    return new MaxPoolingFeedForwardLayerImpl(primaryAxons.dup());
  }
}
