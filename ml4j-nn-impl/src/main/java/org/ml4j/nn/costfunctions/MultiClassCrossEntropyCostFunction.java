package org.ml4j.nn.costfunctions;

import org.ml4j.Matrix;

public class MultiClassCrossEntropyCostFunction {

  public double getCost(Matrix desiredOutputs, Matrix actualOutputs) {
    int m = desiredOutputs.getRows();

    Matrix J_part = (desiredOutputs.mul(-1).mul(limitLog(actualOutputs))).rowSums();

    return J_part.sum() / (2 * m);

  }

  private double limit(double p) {
    p = Math.min(p, 1 - 0.000000000000001);
    p = Math.max(p, 0.000000000000001);
    return p;
  }

  private Matrix limitLog(Matrix m) {
    Matrix x = m.dup();
    for (int i = 0; i < x.getLength(); i++)
      x.put(i, (double) Math.log(limit(x.get(i))));
    return x;
  }
}
