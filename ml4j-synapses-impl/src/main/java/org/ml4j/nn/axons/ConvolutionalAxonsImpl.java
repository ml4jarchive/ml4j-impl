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

  private Map<Integer, WeightIndex[][]> sharedValueListsForOutputChannel;
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory) {
    super(leftNeurons, rightNeurons, matrixFactory);
    validate();
  }

  public ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      MatrixFactory matrixFactory, Matrix connectionWeights) {
    super(leftNeurons, rightNeurons, matrixFactory, connectionWeights);
    validate();
  }

  protected ConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
      Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask);
    validate();
  }
  
  private void validate() {
    int inputWidth = getLeftNeurons().getWidth();
    int outputWidth = getRightNeurons().getWidth();

    int filterWidth = inputWidth + (1 - outputWidth) * (getStride());
    if (filterWidth  <= 0) {
      throw new IllegalStateException("Filter width must be greater than 0");
    }
    int validOutputWidth =
        (int) (((double) (inputWidth - filterWidth)) / ((double) getStride())) + 1;

    if (validOutputWidth != outputWidth) {
      throw new IllegalStateException("Invalid configuration");
    }
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

   
    Matrix initialWeights =  weights.mul(scalingFactor);
    if (getLeftNeurons().hasBiasUnit()) {
      
      initialWeights.putRow(0, matrixFactory.createZeros(1, initialWeights.getColumns()));
      
    }
    if (getRightNeurons().hasBiasUnit()) {
      initialWeights.putColumn(0, matrixFactory.createZeros(initialWeights.getRows(),1));
    }
    return initialWeights;
  }

  @Override
  public ConvolutionalAxons dup() {
    return new ConvolutionalAxonsImpl(leftNeurons, rightNeurons, connectionWeights.dup(),
        connectionWeightsMask);
  }

  @Override
  protected void applyAdditionalConnectionWeightAdjustmentConstraints(Matrix adjustmentRequest) {
    
    // For each output channel
    for (int outputChannel = 0; outputChannel < getRightNeurons().getDepth(); outputChannel++) {

      // Obtain a list of the shared value matrix indexes for each shared value (parameter)
      WeightIndex[][] sharedValueLists = getSharedValueListsForOutputChannel(outputChannel);

      // The number of shared values (parameters)
      int sharedValueCount = sharedValueLists.length;
      
      // For each parameter 
      for (int sharedValueId = 0; sharedValueId <  sharedValueCount; sharedValueId++) {
        
        // Obtain the list of indexes of all the elements sharing the same parameter value.
        WeightIndex[] sharedValueList = sharedValueLists[sharedValueId];
        
        double total = 0;
        double count = 0;
        
        // Average the values of all these elements.
        for (WeightIndex sharedWeightIndex : sharedValueList) {
          total = total
              + adjustmentRequest.get(sharedWeightIndex.getRow(), sharedWeightIndex.getColumn());
          count++;
        }
        double average = total / count;

        // Set the value of each of these elements to be the average.
        for (WeightIndex sharedWeightIndex : sharedValueList) {            
          adjustmentRequest.put(sharedWeightIndex.getRow(), 
              sharedWeightIndex.getColumn(), average);
        }
      }
    }
  }

  private WeightIndex[][] getSharedValueListsForOutputChannel(int outputChannelIndex) {
    
    if (sharedValueListsForOutputChannel == null) {
      sharedValueListsForOutputChannel = new HashMap<>();
    }
    
    WeightIndex[][] sharedValueLists = sharedValueListsForOutputChannel.get(outputChannelIndex);
    if (sharedValueLists != null) {
      return sharedValueLists;
    }
    
    // int startColumnIndex = outputChannel * filterOutputSize;
    int inputWidth = getLeftNeurons().getWidth();
    int outputWidth = getRightNeurons().getWidth();
    int inputHeight = getLeftNeurons().getHeight();
    int outputHeight = getRightNeurons().getHeight();
    int filterWidth = inputWidth + (1 - outputWidth) * (getStride());
    int filterHeight = inputHeight + (1 - outputHeight) * (getStride());

    // We have a shared value list for each element that our filter spans.
    // This shared value list will contain an entry for every output neuron
    // in the channel.
    int sharedValueListCount = filterWidth * filterHeight * this.getLeftNeurons().getDepth()
        + (leftNeurons.hasBiasUnit() ? (isSharedBias() ? 1 : 0) : 0);

    int outputChannelNeuronCount = getOutputNeuronCount() / this.getRightNeurons().getDepth();

    sharedValueLists =
        new WeightIndex[sharedValueListCount][outputChannelNeuronCount];

    for (int outputChannelNeuronIndex =
        0; outputChannelNeuronIndex < outputChannelNeuronCount; outputChannelNeuronIndex++) {

      int outputNeuronIndex = outputChannelNeuronIndex 
           + outputChannelNeuronCount * outputChannelIndex;

      int[] inputIndexesForOutputNeuron = connectionWeightsMask
          .getUnmaskedInputNeuronIndexesForOutputNeuronIndex(outputNeuronIndex);

      int[] sharedValueIndexesForSpecifiedOutputNeuron = null;
      if (this.getLeftNeurons().hasBiasUnit()) {
        if (!isSharedBias()) {
          sharedValueIndexesForSpecifiedOutputNeuron =
              new int[inputIndexesForOutputNeuron.length - 1];

          int numberOfSharedValues = sharedValueIndexesForSpecifiedOutputNeuron.length;

          for (int sharedValueId = 0; sharedValueId < numberOfSharedValues; sharedValueId++) {
            sharedValueIndexesForSpecifiedOutputNeuron[sharedValueId] =
                inputIndexesForOutputNeuron[sharedValueId + 1];
          }

        } else {
          sharedValueIndexesForSpecifiedOutputNeuron = inputIndexesForOutputNeuron;

        }
      } else {
        sharedValueIndexesForSpecifiedOutputNeuron = inputIndexesForOutputNeuron;
      }

      int numberOfSharedValues = sharedValueIndexesForSpecifiedOutputNeuron.length;

      for (int sharedValueId =
          0; sharedValueId < numberOfSharedValues; sharedValueId++) {

        int sharedValueIndexForSpecifiedOutputNeuronAndSharedValueId =
            sharedValueIndexesForSpecifiedOutputNeuron[sharedValueId];

        sharedValueLists[sharedValueId][outputChannelNeuronIndex] = new WeightIndex(
            sharedValueIndexForSpecifiedOutputNeuronAndSharedValueId, outputNeuronIndex);

      }
    }
    sharedValueListsForOutputChannel.put(outputChannelIndex, sharedValueLists);
    return sharedValueLists;
  }


  public int getInputSynapseCount() {
    return getLeftNeurons().getNeuronCountIncludingBias();
  }

  public int getOutputSynapseCount() {
    return getRightNeurons().getNeuronCountIncludingBias();
  }

  public int getOutputNeuronCount() {
    return getRightNeurons().getNeuronCountIncludingBias();
  }

  public int getInputNeuronCount() {
    return getLeftNeurons().getNeuronCountIncludingBias();
  }

  @Override
  protected ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory) {
    return new ConvolutionalWeightsMask(matrixFactory, leftNeurons, rightNeurons, getStride(),
        true);
  }

  private int getStride() {
    return 1;
  }
  
  /**
   * 
   * @return Whether the bias (if present) is shared by each application of the filter over the
   *         input volume, or whether each filter application has its own bias.
   */
  private boolean isSharedBias() {
    return false;
  }
  
  
  private class WeightIndex {

    private int row;
    private int column;
    
    /**
     * @param row The row.
     * @param column The column.
     */
    public WeightIndex(int row, int column) {
      super();
      this.row = row;
      this.column = column;
    }

    public int getRow() {
      return row;
    }

    public int getColumn() {
      return column;
    }

    @Override
    public String toString() {
      return row + "-" + column;
    }
  }
}
