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
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base Axons implementation.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the left hand side of these Axons
 * @param <R> The type of Neurons on the right hand side of these Axons
 * @param <A> The type of these Axons
 */
public abstract class AxonsBase<L extends Neurons, 
    R extends Neurons, A extends Axons<L, R, A>> implements Axons<L, R, A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(
      AxonsBase.class);
  
  protected L leftNeurons;
  protected R rightNeurons;
  protected MatrixFactory matrixFactory;
  protected Matrix connectionWeights;
  
  /**
   * Construct a new Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights Matrix
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory,
      Matrix connectionWeights) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.matrixFactory = matrixFactory;
    this.connectionWeights = matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias());
    adjustConnectionWeights(connectionWeights, ConnectionWeightsAdjustmentDirection.ADDITION);
  }
  
  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public NeuronsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {
    LOGGER.debug("Pushing left to right through Axons");
    if (leftNeuronsActivation.getFeatureOrientation()
            != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with COLUMNS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    Matrix outputMatrix =
        leftNeuronsActivation.withBiasUnit(leftNeurons.hasBiasUnit(), axonsContext).getActivations()
            .mmul(connectionWeights);
    return new NeuronsActivation(outputMatrix, rightNeurons.hasBiasUnit(),
        leftNeuronsActivation.getFeatureOrientation()).withBiasUnit(rightNeurons.hasBiasUnit(),
            axonsContext);
  }

  @Override
  public NeuronsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    LOGGER.debug("Pushing right to left through Axons:");
    if (rightNeuronsActivation.getFeatureOrientation()
            != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
 
    Matrix outputMatrix =
        connectionWeights.mmul(rightNeuronsActivation.withBiasUnit(rightNeurons.hasBiasUnit(), 
            axonsContext).getActivations());
    return new NeuronsActivation(outputMatrix, leftNeurons.hasBiasUnit(),
        rightNeuronsActivation.getFeatureOrientation()).withBiasUnit(leftNeurons.hasBiasUnit(),
            axonsContext);
  }
  
  @Override
  public Matrix getDetachedConnectionWeights() {
    return connectionWeights.dup();
  }
  
  protected void adjustConnectionWeights(Matrix adjustment,
      ConnectionWeightsAdjustmentDirection adjustmentDirection) {
    if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
      connectionWeights.addi(adjustment);
    } else {
      connectionWeights.subi(adjustment);
    }
  }
}
