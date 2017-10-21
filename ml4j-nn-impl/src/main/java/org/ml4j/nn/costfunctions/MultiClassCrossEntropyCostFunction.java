package org.ml4j.nn.costfunctions;

import org.ml4j.Matrix;

public class MultiClassCrossEntropyCostFunction {

  /**
   * 
   * @param desiredOutputs The desired outputs.
   * @param actualOutputs The actual outputs.
   * @return The cost associated with producing the actual outputs given the desired outputs.
   */
  public double getCost(Matrix desiredOutputs, Matrix actualOutputs) {
    
    int count = desiredOutputs.getRows();

    Matrix jpart = (desiredOutputs.mul(-1).mul(limitLog(actualOutputs))).rowSums();

    return jpart.sum() / (2 * count);

  }

  private double limit(double value) {
    value = Math.min(value, 1 - 0.000000000000001);
    value = Math.max(value, 0.000000000000001);
    return value;
  }

  private Matrix limitLog(Matrix matrix) {
    Matrix dupMatrix = matrix.dup();
    for (int i = 0; i < dupMatrix.getLength(); i++) {
      dupMatrix.put(i, (double) Math.log(limit(dupMatrix.get(i))));
    }
    return dupMatrix;
  }
}
