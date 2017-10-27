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
import org.ml4j.nn.neurons.Neurons3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of AveragePoolingAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class AveragePoolingAxonsImpl 
    extends AxonsBase<Neurons3D, Neurons3D, AveragePoolingAxons>
    implements AveragePoolingAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      AveragePoolingAxonsImpl.class);
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public AveragePoolingAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory) {
    super(leftNeurons, rightNeurons, matrixFactory, 
        createAxonConnectionWeights(leftNeurons, rightNeurons, matrixFactory));
  }
 
  
  /**
   * Obtain the axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  private static Matrix createAxonConnectionWeights(Neurons inputNeurons,
      Neurons outputNeurons, MatrixFactory matrixFactory) {
    LOGGER.debug("Initialising Average Pooling weights...");
    throw new UnsupportedOperationException("Not yet implemented");
  }

  @Override
  public AveragePoolingAxons dup() {
    return new AveragePoolingAxonsImpl(leftNeurons, rightNeurons, matrixFactory);
  }
}
