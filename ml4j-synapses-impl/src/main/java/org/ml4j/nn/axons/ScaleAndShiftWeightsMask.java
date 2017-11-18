package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;

public class ScaleAndShiftWeightsMask implements ConnectionWeightsMask {

  private Matrix weightsMask;
  private boolean rightNeuronsBiasUnit;

  /**
   * Create a new ScaleAndShiftWeightsMask for these neurons.
   * 
   * @param matrixFactory The matrix factory.
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   */
  public ScaleAndShiftWeightsMask(MatrixFactory matrixFactory, Neurons leftNeurons,
      Neurons rightNeurons) {

    if (!leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException("Left neurons should have a bias unit in order to shift");
    }
    if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException(
          "Left and right neuron counts excluding bias should be the same");
    }
    this.weightsMask = matrixFactory.createMatrix(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias());

    if (leftNeurons.hasBiasUnit()) {

      weightsMask.putRow(0, matrixFactory.createOnes(1, weightsMask.getColumns()));
    }
    if (rightNeurons.hasBiasUnit()) {
      rightNeuronsBiasUnit = true;
      weightsMask.putColumn(0, matrixFactory.createZeros(weightsMask.getRows(), 1));
    }
    for (int i = 0; i < weightsMask.getColumns(); i++) {
      weightsMask.put(i + 1, i, 1);
    }
  }


  @Override
  public int[] getUnmaskedInputNeuronIndexesForOutputNeuronIndex(int outputNeuronIndex) {
    if (rightNeuronsBiasUnit && outputNeuronIndex == 0) {
      return new int[0];
    }
    int[] indexes = new int[2];
    indexes[0] = 1;
    indexes[1] = outputNeuronIndex + 1;
    return indexes;
  }

  @Override
  public Matrix getWeightsMask() {
    return weightsMask; 
  }

}
