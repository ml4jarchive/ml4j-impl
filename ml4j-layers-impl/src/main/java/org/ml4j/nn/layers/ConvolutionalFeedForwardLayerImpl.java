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

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.ConvolutionalAxonsImpl;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of a ConvolutionalFeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class ConvolutionalFeedForwardLayerImpl 
    extends FeedForwardLayerBase<ConvolutionalAxons, ConvolutionalFeedForwardLayer>
    implements ConvolutionalFeedForwardLayer {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  public ConvolutionalFeedForwardLayerImpl(ConvolutionalAxons primaryAxons,
      DifferentiableActivationFunction activationFunction) {
    super(primaryAxons, activationFunction);
  }
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param primaryActivationFunction The primary activation function.
   * @param matrixFactory The MatrixFactory to use to initialise the weights
   */
  public ConvolutionalFeedForwardLayerImpl(Neurons3D inputNeurons, Neurons3D outputNeurons,
      DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory) {
    super(
        new ConvolutionalAxonsImpl(inputNeurons, outputNeurons,
            matrixFactory),
        primaryActivationFunction);
  }
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param primaryActivationFunction The primary activation function.
   * @param matrixFactory The MatrixFactory to use to initialise the weights
   * @param connectionWeights The connectionWeights
   */
  public ConvolutionalFeedForwardLayerImpl(Neurons3D inputNeurons, Neurons3D outputNeurons,
      DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory,
      Matrix connectionWeights) {
    super(
        new ConvolutionalAxonsImpl(inputNeurons, outputNeurons,
            matrixFactory, connectionWeights),
        primaryActivationFunction);
  }

  @Override
  public ConvolutionalFeedForwardLayer dup() {
    return new ConvolutionalFeedForwardLayerImpl(primaryAxons.dup(), primaryActivationFunction);
  }
}
