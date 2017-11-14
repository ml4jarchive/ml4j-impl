package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class ConvolutionalWeightsMask implements ConnectionWeightsMask {

  private static final Logger LOGGER = LoggerFactory.getLogger(ConvolutionalWeightsMask.class);
  
  private Matrix connectionWeightsMask;
  private Map<Integer, int[]> unmaskedInputNeuronIndexesByOutputNeuronIndex;

  /**
   * @param matrixFactory The matrix factory.
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param stride The stride;
   * @param independentInputOutputChannels Whether the input and output channels are independent of
   *        each other.
   */
  public ConvolutionalWeightsMask(MatrixFactory matrixFactory, Neurons3D leftNeurons,
      Neurons3D rightNeurons, int stride, boolean independentInputOutputChannels) {
    this.connectionWeightsMask = createWeightsMask(matrixFactory, leftNeurons, rightNeurons, stride,
        independentInputOutputChannels);
    this.unmaskedInputNeuronIndexesByOutputNeuronIndex = new HashMap<>();
  }

  @Override
  public Matrix getWeightsMask() {
    return connectionWeightsMask;
  }

  public int getOutputNeuronCount() {
    return connectionWeightsMask.getColumns();
  }
  
  public int getInputNeuronCount() {
    return connectionWeightsMask.getRows();
  }

  @Override
  public int[] getUnmaskedInputNeuronIndexesForOutputNeuronIndex(int outputNeuronIndex) {

    int[] unmaskedInputNeuronIndexes =
        unmaskedInputNeuronIndexesByOutputNeuronIndex.get(outputNeuronIndex);

    if (unmaskedInputNeuronIndexes != null) {
      return unmaskedInputNeuronIndexes;
    } else {
      int[] inds = connectionWeightsMask.getColumn(outputNeuronIndex).findIndices();
      unmaskedInputNeuronIndexesByOutputNeuronIndex.put(outputNeuronIndex, inds);
      return inds;
    }
  }

  private Matrix createWeightsMask(MatrixFactory matrixFactory, Neurons3D leftNeurons,
      Neurons3D rightNeurons, int stride, boolean independentInputOutputChannels) {

    LOGGER.debug("Creating convolutional weights mask");
    
    Matrix thetasMask = matrixFactory.createZeros(leftNeurons.getNeuronCountExcludingBias(),
        rightNeurons.getNeuronCountExcludingBias());
    if (leftNeurons.hasBiasUnit()) {
      thetasMask =
          matrixFactory.createOnes(1, thetasMask.getColumns()).appendVertically(thetasMask);
    }

    if (rightNeurons.hasBiasUnit()) {
      thetasMask = matrixFactory.createOnes(thetasMask.getRows(), 1).appendHorizontally(thetasMask);
    }

    int outputWidth = rightNeurons.getWidth();
    int outputHeight = rightNeurons.getHeight();
    int inputWidth = leftNeurons.getWidth();
    int inputHeight = leftNeurons.getHeight();
    
    int filterWidth = inputWidth + (1 - outputWidth) * (stride);
    int filterHeight = inputHeight + (1 - outputHeight) * (stride);

    int gridInputSize = leftNeurons.getNeuronCountExcludingBias() / leftNeurons.getDepth();
    int filterOutputSize = rightNeurons.getNeuronCountExcludingBias() / rightNeurons.getDepth();

    int strideAmount = stride;

    // For each output channel
    for (int f = 0; f < rightNeurons.getDepth(); f++) {
      // Iterate through each element of output channel ( each row, and column)
      for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
          // Calculate the index of the output Neuron
          int outputInd =
              (filterOutputSize * f) + (i * outputWidth + j) + (rightNeurons.hasBiasUnit() ? 1 : 0);
          // For each input channel

          // Iterate all the rows and columns of the input that this output Neuron applies to
          for (int r = i * strideAmount; r < i * strideAmount + filterHeight; r++) {
            for (int c = j * strideAmount; c < j * strideAmount + filterWidth; c++) {

              if (!independentInputOutputChannels) {

                int grid = f;

                // Calculate the index of the Neuron for this input channel, row and column
                int inputInd =
                    grid * gridInputSize + r * inputWidth + c + (leftNeurons.hasBiasUnit() ? 1 : 0);

                // Put an entry in the weights mask for this input Neuron index and output Neuron
                // index.
                thetasMask.put(inputInd, outputInd, 1);

              } else {
                for (int grid = 0; grid < leftNeurons.getDepth(); grid++) {

                  // Calculate the index of the Neuron for this input channel, row and column
                  int inputInd = grid * gridInputSize + r * inputWidth + c
                      + (leftNeurons.hasBiasUnit() ? 1 : 0);

                  // Put an entry in the weights mask for this input Neuron index and output Neuron
                  // index.
                  thetasMask.put(inputInd, outputInd, 1);
                }
              }
            }
          }
        }
      }
    }
    return thetasMask;
  }
}
