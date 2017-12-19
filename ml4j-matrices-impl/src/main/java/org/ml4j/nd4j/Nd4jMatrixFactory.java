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

package org.ml4j.nd4j;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasMatrixFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Default Nd4j MatrixFactory.
 * 
 * @author Michael Lavelle
 */
public class Nd4jMatrixFactory implements MatrixFactory {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  @Override
  public Matrix createOnes(int rows, int columns) {
    return createNd4jMatrix(Nd4j.ones(rows, columns));
  }

  @Override
  public Matrix createOnes(int length) {
    return createNd4jMatrix(Nd4j.ones(length));
  }

  @Override
  public Matrix createMatrix(double[][] data) {
    return createNd4jMatrix(Nd4j.create(new JBlasMatrixFactory().createMatrix(data)
        .toArray(),
        new int[] {data.length, data[0].length}));
  }

  @Override
  public Matrix createMatrix() {
    return createNd4jMatrix(Nd4j.create());
  }

  @Override
  public Matrix createMatrix(double[] data) {
    return createNd4jMatrix(Nd4j.create(data));
  }

  @Override
  public Matrix createMatrix(int rows, int cols, double[] data) {
    return createNd4jMatrix(Nd4j.create(data, new int[] {rows, cols}));
  }

  @Override
  public Matrix createMatrix(int rows, int cols) {
    return createNd4jMatrix(Nd4j.create(new int[] {rows, cols}));
  }

  @Override
  public Matrix createZeros(int rows, int cols) {
    return createNd4jMatrix(Nd4j.create(new int[] {rows, cols}));
  }

  @Override
  public Matrix createRandn(int rows, int cols) {
    return createNd4jMatrix(Nd4j.randn(rows, cols));
  }

  @Override
  public Matrix createHorizontalConcatenation(Matrix first, Matrix second) {
    return first.appendHorizontally(second);
  }

  @Override
  public Matrix createRand(int rows, int cols) {
    return createNd4jMatrix(Nd4j.rand(rows, cols));
  }

  @Override
  public Matrix createVerticalConcatenation(Matrix first, Matrix second) {
    return first.appendVertically(second);
  }

  protected Nd4jMatrix createNd4jMatrix(INDArray matrix) {
    return new Nd4jMatrix(matrix);
  }
}
