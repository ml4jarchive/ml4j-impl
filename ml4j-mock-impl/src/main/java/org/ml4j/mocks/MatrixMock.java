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

package org.ml4j.mocks;

import org.ml4j.Matrix;

/**
 * A skeleton mock Matrix for structural purposes.
 * 
 * @author Michael Lavelle
 */
public class MatrixMock implements Matrix {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  /**
   * Number of rows in the Matrix.
   */
  private int rows;

  /**
   * Number of columns in the Matrix.
   */
  private int columns;


  private double[][] data;

  /**
   * Construct a new MatrixMock in the specified shape.
   * 
   * @param data The data array with first dimension of rows and second dimension of columns
   */
  public MatrixMock(double[][] data) {
    this.rows = data.length;
    this.columns = data[0].length;
    this.data = data;
  }

  @Override
  public int getRows() {
    return rows;
  }

  @Override
  public int getColumns() {
    return columns;
  }

  @Override
  public double[] toArray() {
    double[] dataArray = new double[rows * columns];
    int index = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        dataArray[index++] = data[r][c];
      }
    }
    return dataArray;
  }

  @Override
  public Matrix getRow(int rowIndex) {
    return new MatrixMock(new double[][] {data[rowIndex] });
  }

  @Override
  public Matrix getColumn(int columnIndex) {
    throw new UnsupportedOperationException("Not yet implemented");
  }
}
