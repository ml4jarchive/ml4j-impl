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
        Matrix connectionWeights, Matrix connectionWeightsMask) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask);
  }

  @Override
  protected Matrix createConnectionWeightsMask(MatrixFactory matrixFactory) {
    LOGGER.debug("Creating Pooling Connection Weights Mask");
    
    int inputNeuronCount = getLeftNeurons().getNeuronCountExcludingBias();
    int outputNeuronCount = getRightNeurons().getNeuronCountExcludingBias();

    int depth = getLeftNeurons().getDepth();
    
    Matrix thetasMask = matrixFactory.createMatrix(inputNeuronCount, outputNeuronCount);

    int outputDim = (int) Math.sqrt(outputNeuronCount / depth);
    int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

    int gridInputSize = inputNeuronCount / depth;
    int gridOutputSize = outputNeuronCount / depth;

    int scale = inputDim / outputDim;

    for (int grid = 0; grid < depth; grid++) {
      for (int i = 0; i < outputDim; i++) {
        for (int j = 0; j < outputDim; j++) {

          int startInputRow = i * scale;
          int startInputCol = j * scale;
          int outputInd = grid * gridOutputSize + (i * outputDim + j);
          for (int r = 0; r < scale; r++) {
            for (int c = 0; c < scale; c++) {
              int row = startInputRow + r;
              int col = startInputCol + c;
              int inputInd = grid * gridInputSize + row * inputDim + col;
              thetasMask.put(inputInd, outputInd, 1);

            }
          }
        }
      }
    }
    
    if (getLeftNeurons().hasBiasUnit()) {
      thetasMask =
          matrixFactory.createZeros(1, thetasMask.getColumns()).appendVertically(thetasMask);
    }
    if (getRightNeurons().hasBiasUnit()) {
      thetasMask =
          matrixFactory.createZeros(thetasMask.getRows(), 1).appendHorizontally(thetasMask);
    }
        
    return thetasMask;
  }
}
