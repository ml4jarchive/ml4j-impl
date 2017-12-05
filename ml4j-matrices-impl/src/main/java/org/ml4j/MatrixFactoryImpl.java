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

package org.ml4j;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

/**
 * A skeleton MatrixFactory impl.
 * 
 * @author Michael Lavelle
 */
public class MatrixFactoryImpl implements MatrixFactory {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  @Override
  public Matrix createOnes(int rows, int columns) {
    double[][] data = new double[rows][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        data[r][c] = 1;
      }
    }
    return createMatrix(data);
  }
  
  @Override
  public Matrix createOnes(int arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }


  @Override
  public Matrix createMatrix(double[][] data) {
    return new MatrixImpl(data);
  }
  
  @Override
  public Matrix createMatrix() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createMatrix(double[] arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createMatrix(int arg0, int arg1, double[] arg2) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createMatrix(int arg0, int arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createZeros(int rows, int columns) {
    return new MatrixImpl(new double[rows][columns]);
  }

  @Override
  public Matrix createRandn(int rows, int columns) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createHorizontalConcatenation(Matrix arg0, Matrix arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
 
  @Override
  public Matrix createRand(int arg0, int arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createVerticalConcatenation(Matrix arg0, Matrix arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
