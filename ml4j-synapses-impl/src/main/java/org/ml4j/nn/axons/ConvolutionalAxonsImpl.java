/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Default implementation of ConvolutionalAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ConvolutionalAxonsImpl extends
    TrainableAxonsBase<Neurons3D, Neurons3D, ConvolutionalAxons> implements ConvolutionalAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(ConvolutionalAxonsImpl.class);

  private boolean sameBiasWithinEachFilter;
  private Map<Integer, int[][]> sharedValueIndexesByFilterIndex;


  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory) {
    super(leftNeurons, rightNeurons, matrixFactory);
    this.sharedValueIndexesByFilterIndex = new HashMap<Integer, int[][]>();
  }
  
  public ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory, Matrix connectionWeights) {
    super(leftNeurons, rightNeurons, matrixFactory, connectionWeights);
    this.sharedValueIndexesByFilterIndex = new HashMap<Integer, int[][]>();
  }
  
  protected ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons, 
        Matrix connectionWeights, Matrix connectionWeightsMask) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask);
    this.sharedValueIndexesByFilterIndex = new HashMap<Integer, int[][]>();
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

    LOGGER.debug("Initialising FullyConnectedAxon weights...");

    Matrix weights = matrixFactory.createRandn(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias());

    double scalingFactor = Math.sqrt(2d / ((double) leftNeurons.getNeuronCountIncludingBias()));

    return weights.mul(scalingFactor);
  }

  @Override
  public ConvolutionalAxons dup() {
    return new ConvolutionalAxonsImpl(leftNeurons, rightNeurons, 
        connectionWeights, connectionWeightsMask);
  }
 
  @Override
  protected void applyAdditionalConnectionWeightAdjustmentConstraints(Matrix adjustmentRequest) {
    
    
    boolean hasBiasUnit = this.getLeftNeurons().hasBiasUnit();

    int inputWidth = (int) Math.sqrt(getInputSynapseCount() / getDepth());
    int outputWidth = (int) Math.sqrt(getOutputSynapseCount() / getFilterCount());
    int filterWidth = inputWidth + (1 - outputWidth) * (getStride());
    int sharedValueCount =
        filterWidth * filterWidth * getDepth() + (hasBiasUnit && sameBiasWithinEachFilter ? 1 : 0);

    int filterOutputSize = getRightNeurons().getNeuronCountExcludingBias() / getFilterCount();

    for (int f = 0; f < getFilterCount(); f++) {

      int startColumnIndex = f * filterOutputSize + (this.getRightNeurons().hasBiasUnit() ? 1 : 0);

      int[][] sharedValueIndexes = getSharedValueIndexes(f);
      double[] averageValues = new double[sharedValueCount];
      for (int column = 0; column < filterOutputSize; column++) {
        int[] inds = getUnmaskedInputNeuronIndexesForOutputNeuronIndex(column + startColumnIndex);
        sharedValueIndexes[column] = inds;
        for (int i = 0; i < averageValues.length; i++) {
          averageValues[i] = averageValues[i]
              + adjustmentRequest.get(inds[+i + (hasBiasUnit && !sameBiasWithinEachFilter ? 1 : 0)],
                  column + startColumnIndex);
        }
      }

      for (int i = 0; i < averageValues.length; i++) {
        averageValues[i] = averageValues[i] / filterOutputSize;
      }
      for (int column = 0; column < filterOutputSize; column++) {
        for (int sharedValueIndex = 0; sharedValueIndex < sharedValueCount; sharedValueIndex++) {
          adjustmentRequest.put(
              sharedValueIndexes[column][sharedValueIndex
                  + +(hasBiasUnit && !sameBiasWithinEachFilter ? 1 : 0)],
              column + startColumnIndex, averageValues[sharedValueIndex]);
        }
      }
    }
  }

  private int[][] getSharedValueIndexes(int index) {

    if (sharedValueIndexesByFilterIndex == null) {
      sharedValueIndexesByFilterIndex = new HashMap<>();
    }
    
    int[][] indexes = sharedValueIndexesByFilterIndex.get(index);
    if (indexes != null) {
      return indexes;
    }

    int filterOutputSize = getRightNeurons().getNeuronCountExcludingBias() / getFilterCount();

    int startColumnIndex =
        index * filterOutputSize + (this.getRightNeurons().hasBiasUnit() ? 1 : 0);

    boolean hasBiasUnit = this.getLeftNeurons().hasBiasUnit();

    int inputWidth = (int) Math.sqrt(getInputSynapseCount() / getDepth());
    int outputWidth = (int) Math.sqrt(getOutputSynapseCount() / getFilterCount());


    int filterWidth = inputWidth + (1 - outputWidth) * (getStride());


    int sharedValueCount =
        filterWidth * filterWidth * getDepth() + (hasBiasUnit && sameBiasWithinEachFilter ? 1 : 0);

    int[][] sharedValueIndexes = new int[filterOutputSize][sharedValueCount
        + (hasBiasUnit && !sameBiasWithinEachFilter ? 1 : 0)];
    for (int column = 0; column < filterOutputSize; column++) {
      int[] inds = getUnmaskedInputNeuronIndexesForOutputNeuronIndex(column + startColumnIndex);
      sharedValueIndexes[column] = inds;

    }

    sharedValueIndexesByFilterIndex.put(index, sharedValueIndexes);

    return sharedValueIndexes;
  }

  public int getInputSynapseCount() {
    return getLeftNeurons().getNeuronCountIncludingBias();
  }

  public int getOutputSynapseCount() {
    return getRightNeurons().getNeuronCountIncludingBias();
  }

  private int getFilterCount() {
    return getRightNeurons().getDepth();
  }

  private int getDepth() {
    return getLeftNeurons().getDepth();
  }

  protected int[] getUnmaskedInputNeuronIndexesForOutputNeuronIndex(int column) {

    int[][] inputNeuronsIndexesForOutputNeuronIndex = new int[getOutputNeuronCount()][];


    int[] inds = inputNeuronsIndexesForOutputNeuronIndex[column];

    if (inds == null) {
      inds = this.connectionWeightsMask.getColumn(column).findIndices();
      inputNeuronsIndexesForOutputNeuronIndex[column] = inds;
    }

    return inds;
  }

  public int getOutputNeuronCount() {
    return getRightNeurons().getNeuronCountIncludingBias();
  }

  public int getInputNeuronCount() {
    return getLeftNeurons().getNeuronCountIncludingBias();
  }

  @Override
  protected Matrix createConnectionWeightsMask(MatrixFactory matrixFactory) {

    Matrix thetasMask =
        matrixFactory.createZeros(this.getLeftNeurons().getNeuronCountExcludingBias(),
            this.getRightNeurons().getNeuronCountExcludingBias());
    if (this.getLeftNeurons().hasBiasUnit()) {
      thetasMask =
          matrixFactory.createOnes(1, thetasMask.getColumns()).appendVertically(thetasMask);
    }

    if (this.getRightNeurons().hasBiasUnit()) {
      thetasMask = matrixFactory.createOnes(thetasMask.getRows(), 1).appendHorizontally(thetasMask);
    }

    int outputDim = (int) Math.sqrt(
        this.getRightNeurons().getNeuronCountExcludingBias() / this.getRightNeurons().getDepth());
    int inputDim = (int) Math.sqrt(
        this.getLeftNeurons().getNeuronCountExcludingBias() / this.getLeftNeurons().getDepth());

    int filterWidth = inputDim + (1 - outputDim) * (getStride());
    int gridInputSize =
        this.getLeftNeurons().getNeuronCountExcludingBias() / this.getLeftNeurons().getDepth();
    int filterOutputSize =
        this.getRightNeurons().getNeuronCountExcludingBias() / this.getRightNeurons().getDepth();

    int strideAmount = getStride();

    for (int grid = 0; grid < this.getLeftNeurons().getDepth(); grid++) {
      for (int f = 0; f < this.getRightNeurons().getDepth(); f++) {
        for (int i = 0; i < outputDim; i++) {
          for (int j = 0; j < outputDim; j++) {
            for (int r = i * strideAmount; r < i * strideAmount + filterWidth; r++) {
              for (int c = j * strideAmount; c < j * strideAmount + filterWidth; c++) {
                int outputInd = (filterOutputSize * f) + (i * outputDim + j)
                    + (this.getRightNeurons().hasBiasUnit() ? 1 : 0);;
                int inputInd = grid * gridInputSize + r * inputDim + c
                    + (this.getLeftNeurons().hasBiasUnit() ? 1 : 0);

                thetasMask.put(inputInd, outputInd, 1);
              }
            }
          }
        }
      }
    }
    return thetasMask;
  }

  private int getStride() {
    return 1;
  }
}
