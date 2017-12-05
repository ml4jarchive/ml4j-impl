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
import org.jblas.MatrixFunctions;
import org.ml4j.Matrix;

/**
 * Default JBlas matrix implementation.
 * 
 * @author Michael Lavelle
 */
public class JBlasMatrix implements Matrix {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  public DoubleMatrix matrix;

  public JBlasMatrix(DoubleMatrix matrix) {
    this.matrix = matrix;
  }

  private DoubleMatrix getDoubleMatrix() {
    return matrix;
  }

  /**
   * Create a new JBlas DoubleMatrix from the Matrix.
   * 
   * @param matrix The matrix we want to convert to a DoubleMatrix.
   * @return The resulting DoubleMatrix.
   */
  private DoubleMatrix createJBlasDoubleMatrix(Matrix matrix) {
    if (matrix instanceof JBlasMatrix) {
      return ((JBlasMatrix) matrix).getDoubleMatrix();
    } else {
      return new DoubleMatrix(matrix.toArray2());
    }
  }

  protected JBlasMatrix createJBlasMatrix(DoubleMatrix matrix) {
    return new JBlasMatrix(matrix);
  }

  @Override
  public Matrix add(Matrix other) {
    return createJBlasMatrix(matrix.add(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix add(double value) {
    return createJBlasMatrix(matrix.add(value));
  }

  @Override
  public Matrix addi(Matrix other) {
    matrix.addi(createJBlasDoubleMatrix(other));
    return this;
  }

  @Override
  public Matrix addi(double value) {
    matrix.addi(value);
    return this;
  }

  @Override
  public int argmax() {
    return matrix.argmax();
  }

  @Override
  public Matrix copy(Matrix other) {
    return createJBlasMatrix(matrix.copy(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix div(double value) {
    return createJBlasMatrix(matrix.div(value));
  }
  
  @Override
  public Matrix div(Matrix other) {
    return createJBlasMatrix(matrix.div(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix divi(double value) {
    matrix.divi(value);
    return this;
  }
  
  @Override
  public Matrix divi(Matrix other) {
    matrix.divi(createJBlasDoubleMatrix(other));
    return this;
  }

  @Override
  public Matrix diviColumnVector(Matrix other) {
    matrix.diviColumnVector(createJBlasDoubleMatrix(other));
    return this;
  }

  @Override
  public double dot(Matrix other) {
    return matrix.dot(createJBlasDoubleMatrix(other));
  }

  @Override
  public int[] findIndices() {
    return matrix.findIndices();
  }

  @Override
  public double get(int index) {
    return matrix.get(index);
  }
  
  @Override
  public double get(int row, int col) {
    return matrix.get(row, col);
  }

  @Override
  public Matrix get(int[] rows, int[] cols) {
    return createJBlasMatrix(matrix.get(rows, cols));
  }
  
  @Override
  public int getColumns() {
    return matrix.getColumns();
  }

  @Override
  public Matrix getColumns(int[] cols) {
    return createJBlasMatrix(matrix.getColumns(cols));
  }

  @Override
  public int getLength() {
    return matrix.getLength();
  }

  @Override
  public Matrix getRowRange(int avalue, int bvalue, int cvalue) {
    return createJBlasMatrix(matrix.getRowRange(avalue, bvalue, cvalue));
  }
  
  @Override
  public int getRows() {
    return matrix.getRows();
  }

  @Override
  public Matrix getRows(int[] rows) {
    return createJBlasMatrix(matrix.getRows(rows));
  }

  @Override
  public Matrix mmul(Matrix other, Matrix target) {
    return createJBlasMatrix(
        matrix.mmuli(createJBlasDoubleMatrix(other), createJBlasDoubleMatrix(target)));
  }

  @Override
  public Matrix mmul(Matrix other) {
    return createJBlasMatrix(matrix.mmul(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix mul(double value) {
    return createJBlasMatrix(matrix.mul(value));
  }

  @Override
  public Matrix mul(Matrix other) {
    return createJBlasMatrix(matrix.mul(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix muli(Matrix other) {
    matrix.muli(createJBlasDoubleMatrix(other));
    return this;
  }

  @Override
  public Matrix muli(double value) {
    matrix.muli(value);
    return this;
  }

  @Override
  public void put(int index, double value) {
    matrix.put(index, value);
  }

  @Override
  public void put(int row, int col, double value) {
    matrix.put(row, col, value);
  }

  @Override
  public void put(int[] indices, int cvalue, Matrix other) {
    matrix.put(indices, cvalue, createJBlasDoubleMatrix(other));
  }

  @Override
  public void putColumn(int columnIndex, Matrix other) {
    matrix.putColumn(columnIndex, createJBlasDoubleMatrix(other));
  }

  @Override
  public void putRow(int rowIndex, Matrix other) {
    matrix.putRow(rowIndex, createJBlasDoubleMatrix(other));
  }

  @Override
  public void reshape(int newRows, int newColumns) {
    matrix.reshape(newRows, newColumns);
  }

  @Override
  public int[] rowArgmaxs() {
    return matrix.rowArgmaxs();
  }

  @Override
  public Matrix rowSums() {
    return createJBlasMatrix(matrix.rowSums());
  }

  @Override
  public Matrix sub(Matrix other) {
    return createJBlasMatrix(matrix.sub(createJBlasDoubleMatrix(other)));
  }

  @Override
  public Matrix subi(Matrix other) {
    matrix.sub(createJBlasDoubleMatrix(other));
    return this;
  }

  @Override
  public double sum() {
    return matrix.sum();
  }

  @Override
  public double[][] toArray2() {
    return matrix.toArray2();
  }

  @Override
  public Matrix transpose() {
    return createJBlasMatrix(matrix.transpose());
  }

  @Override
  public Matrix appendHorizontally(Matrix other) {
    DoubleMatrix result = DoubleMatrix.concatHorizontally(matrix, createJBlasDoubleMatrix(other));
    return createJBlasMatrix(result);
  }

  @Override
  public Matrix appendVertically(Matrix other) {
    DoubleMatrix result = DoubleMatrix.concatVertically(matrix, createJBlasDoubleMatrix(other));
    return createJBlasMatrix(result);
  }

  @Override
  public Matrix asCudaMatrix() {
    throw new UnsupportedOperationException("CUDA not yet supported");
  }

  @Override
  public Matrix asJBlasMatrix() {
    return this;
  }

  @Override
  public Matrix dup() {
    return createJBlasMatrix(matrix.dup());
  }

  @Override
  public Matrix expi() {
    MatrixFunctions.expi(matrix);
    return this;
  }

  @Override
  public Matrix getColumn(int columnIndex) {
    return createJBlasMatrix(matrix.getColumn(columnIndex));
  }

 
  @Override
  public Matrix getRow(int rowIndex) {
    return createJBlasMatrix(matrix.getRow(rowIndex));
  }


  @Override
  public Matrix log() {
    DoubleMatrix result = MatrixFunctions.log(matrix);
    return createJBlasMatrix(result);
  }

  @Override
  public Matrix logi() {
    MatrixFunctions.logi(matrix);
    return this;
  }


  @Override
  public Matrix pow(int value) {
    DoubleMatrix result = MatrixFunctions.pow(matrix, value);
    return createJBlasMatrix(result);
  }

  @Override
  public Matrix powi(int value) {
    MatrixFunctions.powi(matrix, value);
    return this;
  }

  @Override
  public Matrix sigmoid() {
    DoubleMatrix result = matrix.mul(-1);
    MatrixFunctions.expi(matrix);
    result = result.add(1);
    result = MatrixFunctions.powi(result, -1);
    return this.createJBlasMatrix(result);
  }

  @Override
  public double[] toArray() {
    return matrix.toArray();
  }
}
