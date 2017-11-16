package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;

/**
 * Default base TrainableAxons implementation.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the left hand side of these Axons
 * @param <R> The type of Neurons on the right hand side of these Axons
 * @param <A> The type of these Axons
 */
public abstract class TrainableAxonsBase<L extends Neurons, 
    R extends Neurons, A extends TrainableAxons<L, R, A>, C extends AxonsConfig> 
    extends AxonsBase<L, R, A, C> implements TrainableAxons<L, R, A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  public TrainableAxonsBase(L leftNeurons, R rightNeurons, 
      MatrixFactory matrixFactory, Matrix initialConnectionWeights, 
          ConnectionWeightsMask connectionWeightsMask, C config) {
    super(leftNeurons, rightNeurons, matrixFactory, 
        initialConnectionWeights, connectionWeightsMask, config);
  }
  
  public TrainableAxonsBase(L leftNeurons, R rightNeurons, 
      MatrixFactory matrixFactory, Matrix initialConnectionWeights, C config) {
    super(leftNeurons, rightNeurons, matrixFactory, 
        initialConnectionWeights, config);
  }
  
  public TrainableAxonsBase(L leftNeurons, R rightNeurons, 
      MatrixFactory matrixFactory, C config) {
    super(leftNeurons, rightNeurons, matrixFactory, config);
  }
  
  protected TrainableAxonsBase(L leftNeurons, R rightNeurons, 
      Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask, C config) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask, config);
  }

  @Override
  public void adjustConnectionWeights(Matrix adjustment,
      ConnectionWeightsAdjustmentDirection adjustmentDirection) {
    super.adjustConnectionWeights(adjustment, adjustmentDirection, false);
  }
}
