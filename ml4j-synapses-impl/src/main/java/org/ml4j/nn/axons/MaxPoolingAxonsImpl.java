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
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of MaxPoolingAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class MaxPoolingAxonsImpl 
    extends PoolingAxonsBase<MaxPoolingAxons>
    implements MaxPoolingAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      MaxPoolingAxonsImpl.class);
  
  /**
   * Whether to scale outputs up by a factor equal to the reduction factor due
   * to only outputting max-elements.
   * 
   */
  private boolean scaleOutputs;
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   * @param scaleOutputs Whether to scale outputs up by a factor equal to the 
   *        reduction factor due to only outputting max-elements.
   */
  public MaxPoolingAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory, boolean scaleOutputs) {
    super(leftNeurons, rightNeurons, matrixFactory);
    this.scaleOutputs = scaleOutputs;
    if (leftNeurons.hasBiasUnit()) {
      throw new UnsupportedOperationException("Left neurons with bias not yet supported");
    }
    if (rightNeurons.hasBiasUnit()) {
      throw new UnsupportedOperationException("Left neurons with bias not yet supported");
    }
  }

  protected MaxPoolingAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons, 
        Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask);
  }
  
  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  @Override
  protected Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory) {
    LOGGER.debug("Initialising Max Pooling weights...");
    return matrixFactory.createOnes(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias()).mul(connectionWeightsMask
            .getWeightsMask());
  }

  @Override
  public MaxPoolingAxons dup() {
    return new MaxPoolingAxonsImpl(leftNeurons, rightNeurons, 
        connectionWeights, connectionWeightsMask);
  }

  @Override
  protected Matrix createLeftInputDropoutMask(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {
    Matrix defaultDropoutMask =
        super.createLeftInputDropoutMask(leftNeuronsActivation, axonsContext);
    if (defaultDropoutMask != null) {
      throw new IllegalStateException("Max Pooling Axons use a form of input dropout themselves - "
          + "it is not yet possible to combine with another form of input dropout");
    } else {
      return createMaxPoolingDropoutMask(leftNeuronsActivation, axonsContext.getMatrixFactory());
    }
  }

  private Matrix createMaxPoolingDropoutMask(NeuronsActivation leftNeuronsActivation,
      MatrixFactory matrixFactory) {
    Matrix dropoutMask = matrixFactory.createZeros(leftNeuronsActivation.getActivations().getRows(),
        leftNeuronsActivation.getActivations().getColumns());
        
    for (int outputNeuronIndex = 0; outputNeuronIndex < connectionWeightsMask.getWeightsMask()
        .getColumns(); outputNeuronIndex++) {
      int[] unmaskedInputNeuronIndexesForOutputNeuronIndex = connectionWeightsMask
          .getUnmaskedInputNeuronIndexesForOutputNeuronIndex(outputNeuronIndex);
      for (int r = 0; r < leftNeuronsActivation.getActivations().getRows(); r++) {
        Double maxVal = null;
        Integer maxInd = null;
        for (int i = 0; i < unmaskedInputNeuronIndexesForOutputNeuronIndex.length; i++) {
          double val = leftNeuronsActivation.getActivations().get(r,
              unmaskedInputNeuronIndexesForOutputNeuronIndex[i]);
          if (maxVal == null || val > maxVal.doubleValue()) {
            maxInd = unmaskedInputNeuronIndexesForOutputNeuronIndex[i];
            maxVal = val;
          }
        }
        dropoutMask.put(r, maxInd, 1);
      }
    }
    return dropoutMask;
  }
  
  @Override
  protected double getLeftInputPostDropoutScaling(AxonsContext axonsContext) {

    int outputDim =
        (int) Math.sqrt(this.getRightNeurons().getNeuronCountIncludingBias()
            / this.getRightNeurons().getDepth());
    int inputDim = (int) Math.sqrt(
        this.getLeftNeurons().getNeuronCountIncludingBias() / getLeftNeurons().getDepth());
    
    double scaleDown = inputDim / outputDim;
    if (scaleOutputs) {
      return scaleDown *  scaleDown;
    } else {
      return 1d;
    }
  }
}
