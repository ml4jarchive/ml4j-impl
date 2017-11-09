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
  protected Matrix connectionWeights;
  protected ConnectionWeightsMask connectionWeightsMask;
  
  /**
   * Construct a new Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights mask
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory,
      Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask) {
    this(leftNeurons, rightNeurons, matrixFactory
        .createZeros(leftNeurons.getNeuronCountIncludingBias(), 
            rightNeurons.getNeuronCountIncludingBias()), connectionWeightsMask);
    adjustConnectionWeights(connectionWeights, 
        ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }
  
  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = createConnectionWeightsMask(matrixFactory);
    this.connectionWeights = matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(), 
        rightNeurons.getNeuronCountIncludingBias());
    adjustConnectionWeights(createDefaultInitialConnectionWeights(matrixFactory), 
        ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }
  
  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory,
      Matrix connectionWeights) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = createConnectionWeightsMask(matrixFactory);
    this.connectionWeights = matrixFactory.createZeros(
        leftNeurons.getNeuronCountIncludingBias(), rightNeurons.getNeuronCountIncludingBias());
    adjustConnectionWeights(connectionWeights, 
        ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }
  
  /**
   * Construct a new Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights Matrix
   */
  protected AxonsBase(L leftNeurons, R rightNeurons,
      Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = connectionWeightsMask;
    this.connectionWeights = connectionWeights;
  }

  protected abstract ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory);

  protected abstract 
      Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory);

  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
    LOGGER.debug("Pushing left to right through Axons");
    if (leftNeuronsActivation.getFeatureOrientation()
            != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with COLUMNS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    Matrix outputMatrix = null;
    Matrix outputDropoutMask = null;
    
    Matrix inputDropoutMask = createLeftInputDropoutMask(leftNeuronsActivation, axonsContext);
    
    Matrix previousInputDropoutMask = previousRightToLeftActivation == null ? null
        : previousRightToLeftActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null) {
      outputDropoutMask = previousInputDropoutMask.transpose();
    }

    Matrix inputMatrix = null;
    if (inputDropoutMask != null) {
      double postDropoutScaling = getLeftInputPostDropoutScaling(axonsContext);
      if (postDropoutScaling != 1) {
        Matrix preScaling =  leftNeuronsActivation.withBiasUnit(
            leftNeurons.hasBiasUnit(), axonsContext)
            .getActivations().mul(inputDropoutMask);
       
        inputMatrix = preScaling.mul(postDropoutScaling);
        
        if (leftNeuronsActivation.isBiasUnitIncluded() && leftNeuronsActivation
            .getFeatureOrientation() 
            == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
          inputMatrix.putColumn(0,
              preScaling.getColumn(0));
        } else if (leftNeuronsActivation.isBiasUnitIncluded() && leftNeuronsActivation
            .getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
          inputMatrix.putRow(0,
              preScaling.getRow(0));
        }
        outputMatrix = inputMatrix.mmul(connectionWeights);
      } else {
        inputMatrix = leftNeuronsActivation.withBiasUnit(leftNeurons.hasBiasUnit(), axonsContext)
            .getActivations().mul(inputDropoutMask);
        outputMatrix = inputMatrix.mmul(connectionWeights);
      }

    } else {
      inputMatrix = leftNeuronsActivation.withBiasUnit(leftNeurons.hasBiasUnit(), axonsContext)
          .getActivations();
      outputMatrix = inputMatrix.mmul(connectionWeights);
    }
   
    if (outputDropoutMask != null) {
      outputMatrix = outputMatrix.mul(outputDropoutMask);
    }
 
    return new AxonsActivationImpl(inputDropoutMask, 
        new NeuronsActivation(outputMatrix, rightNeurons.hasBiasUnit(),
        leftNeuronsActivation.getFeatureOrientation()).withBiasUnit(rightNeurons.hasBiasUnit(),
            axonsContext));
  }
  
  /**
   * Return the dropout mask for left hand side input.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the left hand side of these Axons
   */
  protected Matrix createLeftInputDropoutMask(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {

    double leftHandInputDropoutKeepProbability =
        axonsContext.getLeftHandInputDropoutKeepProbability();
    if (leftHandInputDropoutKeepProbability == 1) {
      return null;
    } else {

      Matrix dropoutMask = axonsContext.getMatrixFactory().createZeros(
          leftNeuronsActivation.getActivations().getRows(),
          leftNeuronsActivation.getActivations().getColumns());
      int startRow = leftNeuronsActivation.isBiasUnitIncluded() ? 1 : 0;
      for (int i = 0; i < dropoutMask.getRows(); i++) {
        for (int j = 0; j < dropoutMask.getColumns(); j++) {
          if ((i < startRow) || (Math.random() < leftHandInputDropoutKeepProbability)) {
            dropoutMask.put(i, j, 1);
          }
        }
      }
      return dropoutMask;

    }
  }
  
  /**
   * Return the scaling required due to left-hand side input dropout.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected double getLeftInputPostDropoutScaling(AxonsContext axonsContext) {
    double dropoutKeepProbability = 
        axonsContext.getLeftHandInputDropoutKeepProbability();
    if (dropoutKeepProbability == 0) {
      throw new IllegalArgumentException("Dropout keep probability cannot be set to 0");
    }
    return 1d / dropoutKeepProbability;
  }

  /**
   * Return the scaling required due to right-hand side input dropout.
   * This is not yet supported, so we return 1.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected double getRightInputPostDropoutScaling(AxonsContext axonsContext) {
    return 1d;
  }

  /**
   * Return the dropout mask for right hand side input.
   * This is not yet supported, so we return null.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the right hand side of these Axons
   */
  protected Matrix createRightInputDropoutMask(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    return null;
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
    LOGGER.debug("Pushing right to left through Axons:");
    if (rightNeuronsActivation.getFeatureOrientation()
            != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    
    Matrix outputMatrix = null;
    Matrix outputDropoutMask = null;
    
    Matrix inputDropoutMask = createRightInputDropoutMask(rightNeuronsActivation, axonsContext);
    
    Matrix previousInputDropoutMask = previousLeftToRightActivation == null ? null
        : previousLeftToRightActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null) {
      outputDropoutMask = previousInputDropoutMask.transpose();
    }

    Matrix inputMatrix = null;
    if (inputDropoutMask != null) {
      double postDropoutScaling = getRightInputPostDropoutScaling(axonsContext);
      if (postDropoutScaling != 1) {
        inputMatrix = rightNeuronsActivation.withBiasUnit(rightNeurons.hasBiasUnit(), axonsContext)
            .getActivations().mul(inputDropoutMask).mul(postDropoutScaling);
        outputMatrix = connectionWeights
            .mmul(inputMatrix);
      } else {
        inputMatrix = rightNeuronsActivation.withBiasUnit(rightNeurons.hasBiasUnit(), axonsContext)
            .getActivations().mul(inputDropoutMask);
        outputMatrix = connectionWeights
            .mmul(inputMatrix);
      }
    } else {
      inputMatrix = rightNeuronsActivation
          .withBiasUnit(rightNeurons.hasBiasUnit(), axonsContext).getActivations();
      outputMatrix = connectionWeights.mmul(inputMatrix);
    }
    
    if (outputDropoutMask != null) {
      outputMatrix = outputMatrix.mul(outputDropoutMask);
    }
    
    return new AxonsActivationImpl(inputDropoutMask,
        new NeuronsActivation(outputMatrix, leftNeurons.hasBiasUnit(),
            rightNeuronsActivation.getFeatureOrientation()).withBiasUnit(leftNeurons.hasBiasUnit(),
                axonsContext));
  }
  
  @Override
  public Matrix getDetachedConnectionWeights() {
    return connectionWeights.dup();
  }
  
  protected void adjustConnectionWeights(Matrix adjustment,
      ConnectionWeightsAdjustmentDirection adjustmentDirection, boolean initialisation) {
    if (connectionWeightsMask != null) {
      adjustment.muli(connectionWeightsMask.getWeightsMask());
    }
    applyAdditionalConnectionWeightAdjustmentConstraints(adjustment);
    if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
      connectionWeights.addi(adjustment);
    } else {
      connectionWeights.subi(adjustment);
    }
  }

  protected void applyAdditionalConnectionWeightAdjustmentConstraints(Matrix adjustment) {
    // No-op by default
  }
}