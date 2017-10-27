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

package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of FullyConnectedAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class FullyConnectedAxonsImpl 
    extends TrainableAxonsBase<Neurons, Neurons, FullyConnectedAxons>
    implements FullyConnectedAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      FullyConnectedAxonsImpl.class);
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public FullyConnectedAxonsImpl(Neurons leftNeurons, Neurons rightNeurons,
      MatrixFactory matrixFactory) {
    this(leftNeurons, rightNeurons, matrixFactory, 
        createInitialAxonConnectionWeights(leftNeurons, rightNeurons, matrixFactory));
  }
  
  private FullyConnectedAxonsImpl(Neurons leftNeurons, Neurons rightNeurons,
      MatrixFactory matrixFactory, Matrix connectionWeights) {
    super(leftNeurons, rightNeurons, matrixFactory, connectionWeights);
  }
  
  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  private static Matrix createInitialAxonConnectionWeights(Neurons inputNeurons,
      Neurons outputNeurons, MatrixFactory matrixFactory) {
   
    LOGGER.debug("Initialising FullyConnectedAxon weights...");
    
    Matrix weights = matrixFactory.createRandn(inputNeurons.getNeuronCountIncludingBias(),
        outputNeurons.getNeuronCountIncludingBias());

    double scalingFactor = 
        Math.sqrt(2 / ((double)inputNeurons.getNeuronCountIncludingBias()));
    
    return weights.mul(scalingFactor);
  }

  @Override
  public FullyConnectedAxons dup() {
    return new FullyConnectedAxonsImpl(leftNeurons, rightNeurons, matrixFactory, connectionWeights);
  }
}
