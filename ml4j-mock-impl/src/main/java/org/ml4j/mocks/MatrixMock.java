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
  public Matrix getRows(int[] arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getColumns() {
    return columns;
  }
  
  @Override
  public Matrix getColumns(int[] arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
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
    double[][] data = new double[getRows()][1];
    for (int r = 0; r < getRows(); r++) {
      data[r][0] = this.data[r][columnIndex];
    }
    return new MatrixMock(data);
  }

  @Override
  public Matrix sigmoid() {
    double[][] dataArray = new double[rows][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        double exp = Math.exp(data[r][c]);
        dataArray[r][c] = exp / (1 + exp);
      }
    }
    return new MatrixMock(dataArray);
  }

  @Override
  public Matrix mmul(Matrix other) {
    double[][] dataArray = new double[rows][other.getColumns()];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < other.getColumns(); c++) {
        double result = 0;
        double[] thisRow = getRow(r).toArray();
        double[] otherColumn = other.getColumn(c).toArray();
        for (int i = 0; i < thisRow.length; i++) {
          result = result + thisRow[i] * otherColumn[i];
        }
        dataArray[r][c] = result;
      }
    }
    return new MatrixMock(dataArray);
  }
  
  @Override
  public Matrix mmul(Matrix arg0, Matrix arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public double get(int row, int column) {
    return data[row][column];
  }
  
  @Override
  public double get(int arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
  
  @Override
  public Matrix get(int[] arg0, int[] arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix dup() {
    double[][] dataArray = new double[rows][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        dataArray[r][c] = data[r][c];
      }
    }
    return new MatrixMock(dataArray);
  }

  @Override
  public Matrix appendHorizontally(Matrix other) {
    double[][] dataArray = new double[rows][columns + other.getColumns()];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        dataArray[r][c] = data[r][c];
      }
      for (int c = 0; c < other.getColumns(); c++) {
        dataArray[r][c + columns] = other.get(r, c);
      }
    }
    return new MatrixMock(dataArray);
  }

  @Override
  public Matrix appendVertically(Matrix other) {
    double[][] dataArray = new double[rows + other.getRows()][columns];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        dataArray[r][c] = data[r][c];
      }
    }
    for (int r = 0; r < other.getRows(); r++) {
      for (int c = 0; c < columns; c++) {
        dataArray[r + rows][c] = other.get(r, c);
      }
    }
    return new MatrixMock(dataArray);
  }

  @Override
  public Matrix add(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix add(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix addi(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix addi(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int argmax() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix copy(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix div(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix divi(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix diviColumnVector(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public double dot(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int[] findIndices() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getLength() {
    throw new UnsupportedOperationException("Not implemented yet");

  }

  @Override
  public Matrix getRowRange(int arg0, int arg1, int arg2) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix mul(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix mul(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix muli(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix muli(double arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void put(int arg0, double arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void put(int arg0, int arg1, double arg2) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void put(int[] arg0, int arg1, Matrix arg2) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void putColumn(int arg0, Matrix arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void putRow(int arg0, Matrix arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void reshape(int arg0, int arg1) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int[] rowArgmaxs() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix rowSums() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix sub(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix subi(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public double sum() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public double[][] toArray2() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix transpose() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix asCudaMatrix() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix asJBlasMatrix() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix expi() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix log() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix logi() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix pow(int arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix powi(int arg0) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
