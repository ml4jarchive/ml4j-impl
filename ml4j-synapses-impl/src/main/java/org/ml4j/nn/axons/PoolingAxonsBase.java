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
import org.ml4j.nn.neurons.Neurons3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of MaxPoolingAxons.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class PoolingAxonsBase<A extends PoolingAxons<A>> 
    extends AxonsBase<Neurons3D, Neurons3D, A> implements PoolingAxons<A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      PoolingAxonsBase.class);
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public PoolingAxonsBase(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory) {
    super(leftNeurons, rightNeurons, matrixFactory);
  }

  protected PoolingAxonsBase(Neurons3D leftNeurons, Neurons3D rightNeurons, 
        Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask);
  }

  @Override
  protected ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory) {
    LOGGER.debug("Creating Pooling Connection Weights Mask");
    
    int inputDim = leftNeurons.getWidth() * leftNeurons.getHeight();
    int outputDim = rightNeurons.getWidth() * rightNeurons.getHeight();
    
    int scale = inputDim / outputDim;
    
    int stride = (int)(Math.sqrt(scale));
    
    return 
        new ConvolutionalWeightsMask(matrixFactory, 
            leftNeurons, rightNeurons, stride, false);
  }
}
