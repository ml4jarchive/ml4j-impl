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

package org.ml4j.jblas;

import org.jblas.DoubleMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

/**
 * Default JBlas MatrixFactory.
 * 
 * @author Michael Lavelle
 */
public class JBlasMatrixFactory implements MatrixFactory {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  @Override
  public Matrix createOnes(int rows, int columns) {
    return createJBlasMatrix(DoubleMatrix.ones(rows, columns));
  }

  @Override
  public Matrix createOnes(int length) {
    return createJBlasMatrix(DoubleMatrix.ones(length));
  }

  @Override
  public Matrix createMatrix(double[][] data) {
    return createJBlasMatrix(new DoubleMatrix(data));
  }

  @Override
  public Matrix createMatrix() {
    return createJBlasMatrix(new DoubleMatrix());
  }

  @Override
  public Matrix createMatrix(double[] data) {
    return createJBlasMatrix(new DoubleMatrix(data));
  }

  @Override
  public Matrix createMatrix(int rows, int cols, double[] data) {
    return createJBlasMatrix(new DoubleMatrix(rows, cols, data));
  }

  @Override
  public Matrix createMatrix(int rows, int cols) {
    return createJBlasMatrix(DoubleMatrix.zeros(rows, cols));
  }

  @Override
  public Matrix createZeros(int rows, int cols) {
    return createJBlasMatrix(DoubleMatrix.zeros(rows, cols));
  }

  @Override
  public Matrix createRandn(int rows, int cols) {
    return createJBlasMatrix(DoubleMatrix.randn(rows, cols));
  }

  @Override
  public Matrix createHorizontalConcatenation(Matrix first, Matrix second) {
    return first.appendHorizontally(second);
  }

  @Override
  public Matrix createRand(int rows, int cols) {
    return createJBlasMatrix(DoubleMatrix.rand(rows, cols));
  }

  @Override
  public Matrix createVerticalConcatenation(Matrix first, Matrix second) {
    return first.appendVertically(second);
  }

  protected JBlasMatrix createJBlasMatrix(DoubleMatrix matrix) {
    return new JBlasMatrix(matrix);
  }
}
