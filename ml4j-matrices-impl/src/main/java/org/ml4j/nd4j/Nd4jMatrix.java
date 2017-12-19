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

import org.jblas.DoubleMatrix;
import org.ml4j.Matrix;
import org.ml4j.jblas.JBlasMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Default JBlas matrix implementation.
 * 
 * @author Michael Lavelle
 */
public class Nd4jMatrix implements Matrix {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  public INDArray matrix;

  public Nd4jMatrix(INDArray matrix) {
    this.matrix = matrix;
  }

  private INDArray getIndArray() {
    return matrix;
  }

  /**
   * Create a new n4j4j INDArray from the Matrix.
   * 
   * @param matrix The matrix we want to convert to a INDArray.
   * @return The resulting INDArray.
   */
  private INDArray createNd4jIndArray(Matrix matrix) {
    if (matrix instanceof Nd4jMatrix) {
      return ((Nd4jMatrix) matrix).getIndArray();
    } else if (matrix instanceof JBlasMatrix) {
      return Nd4j.create(matrix.transpose().toArray(),
          new int[] {matrix.getRows(), matrix.getColumns()});
    } else {
      throw new UnsupportedOperationException("Only Nd4j and JBlas matrices currently supported");
    }
  }

  protected Nd4jMatrix createNd4jMatrix(INDArray matrix) {
    return new Nd4jMatrix(matrix);
  }

  @Override
  public Matrix add(Matrix other) {
    return createNd4jMatrix(matrix.add(createNd4jIndArray(other)));
  }

  @Override
  public Matrix add(double value) {
    return createNd4jMatrix(matrix.add(value));
  }

  @Override
  public Matrix addi(Matrix other) {
    this.matrix = matrix.addi(createNd4jIndArray(other));
    return this;
  }

  @Override
  public Matrix addi(double value) {
    this.matrix = this.matrix.addi(value);
    return this;
  }

  @Override
  public int argmax() {
    return asJBlasMatrix().argmax();
  }

  @Override
  public Matrix copy(Matrix other) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  @Override
  public Matrix div(double value) {
    return createNd4jMatrix(matrix.div(value));
  }

  @Override
  public Matrix div(Matrix other) {
    return createNd4jMatrix(matrix.div(createNd4jIndArray(other)));
  }

  @Override
  public Matrix divi(double value) {
    this.matrix = matrix.divi(value);
    return this;
  }

  @Override
  public Matrix divi(Matrix other) {
    this.matrix = matrix.divi(createNd4jIndArray(other));
    return this;
  }

  @Override
  public Matrix diviColumnVector(Matrix other) {
    this.matrix = matrix.diviColumnVector(createNd4jIndArray(other));
    return this;
  }

  @Override
  public double dot(Matrix other) {
    return Nd4j.getBlasWrapper().dot(this.matrix, createNd4jIndArray(other));
  }

  @Override
  public int[] findIndices() {
    int len = 0;
    for (int i = 0; i < getLength(); i++) {
      if (get(i) != 0.0) {
        len++;
      }
    }
    int[] indices = new int[len];
    int counter = 0;

    for (int i = 0; i < getLength(); i++) {
      if (get(i) != 0.0) {
        indices[counter++] = i;
      }
    }
    return indices;
  }

  @Override
  public double get(int index) {
    return matrix.getDouble(index);
  }

  @Override
  public double get(int row, int col) {
    return matrix.getDouble(row, col);
  }

  @Override
  public Matrix get(int[] rows, int[] cols) {
    return createNd4jMatrix(createNd4jIndArray(asJBlasMatrix().get(rows, cols)));
  }

  @Override
  public int getColumns() {
    return matrix.columns();
  }

  @Override
  public Matrix getColumns(int[] cols) {
    return createNd4jMatrix(matrix.getColumns(cols));
  }

  @Override
  public int getLength() {
    return matrix.length();
  }

  @Override
  public Matrix getRowRange(int avalue, int bvalue, int cvalue) {
    return createNd4jMatrix(
        createNd4jIndArray(asJBlasMatrix().getRowRange(avalue, bvalue, cvalue)));
  }

  @Override
  public int getRows() {
    return matrix.rows();
  }

  @Override
  public Matrix getRows(int[] rows) {
    return createNd4jMatrix(matrix.getRows(rows));
  }

  @Override
  public Matrix mmul(Matrix other, Matrix target) {
    return createNd4jMatrix(matrix.mmuli(createNd4jIndArray(other), createNd4jIndArray(target)));
  }

  @Override
  public Matrix mmul(Matrix other) {
    return createNd4jMatrix(matrix.mmul(createNd4jIndArray(other)));
  }

  @Override
  public Matrix mul(double value) {
    return createNd4jMatrix(matrix.mul(value));
  }

  @Override
  public Matrix mul(Matrix other) {
    return createNd4jMatrix(matrix.mul(createNd4jIndArray(other)));
  }

  @Override
  public Matrix muli(Matrix other) {
    this.matrix = matrix.muli(createNd4jIndArray(other));
    return this;
  }

  @Override
  public Matrix muli(double value) {
    this.matrix = matrix.muli(value);
    return this;
  }

  @Override
  public void put(int index, double value) {
    matrix.putScalar(index, value);
  }

  @Override
  public void put(int row, int col, double value) {
    matrix.putScalar(row, col, value);
  }

  @Override
  public void put(int[] indices, int cvalue, Matrix other) {
    // matrix.put(indices, cvalue, createNd4jIndArray(other));
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void putColumn(int columnIndex, Matrix other) {
    matrix.putColumn(columnIndex, createNd4jIndArray(other));
  }

  @Override
  public void putRow(int rowIndex, Matrix other) {
    matrix.putRow(rowIndex, createNd4jIndArray(other));
  }

  @Override
  public void reshape(int newRows, int newColumns) {
    this.matrix = matrix.reshape(newRows, newColumns);
  }

  @Override
  public int[] rowArgmaxs() {
    return asJBlasMatrix().rowArgmaxs();
  }

  @Override
  public Matrix rowSums() {
    return asJBlasMatrix().rowSums();
  }

  @Override
  public Matrix sub(Matrix other) {
    return createNd4jMatrix(matrix.sub(createNd4jIndArray(other)));
  }

  @Override
  public Matrix subi(Matrix other) {
    this.matrix = matrix.subi(createNd4jIndArray(other));
    return this;
  }

  @Override
  public double sum() {
    return asJBlasMatrix().sum();
  }

  @Override
  public double[][] toArray2() {
    return asJBlasMatrix().toArray2();
  }

  @Override
  public Matrix transpose() {
    return createNd4jMatrix(matrix.transpose());
  }

  @Override
  public Matrix appendHorizontally(Matrix other) {
    return createNd4jMatrix(Nd4j.hstack(matrix, createNd4jIndArray(other)));
  }

  @Override
  public Matrix appendVertically(Matrix other) {
    return createNd4jMatrix(Nd4j.vstack(matrix, createNd4jIndArray(other)));
  }

  @Override
  public Matrix asCudaMatrix() {
    throw new UnsupportedOperationException("CUDA not yet supported");
  }

  @Override
  public Matrix asJBlasMatrix() {
    return new JBlasMatrix(
        new DoubleMatrix(matrix.rows(), matrix.columns(), matrix.transpose().data().asDouble()));
  }

  @Override
  public Matrix dup() {
    return createNd4jMatrix(matrix.dup());
  }

  @Override
  public Matrix expi() {
    this.matrix = Transforms.exp(matrix);
    return this;
  }

  @Override
  public Matrix getColumn(int columnIndex) {
    return createNd4jMatrix(matrix.getColumn(columnIndex));
  }


  @Override
  public Matrix getRow(int rowIndex) {
    return createNd4jMatrix(matrix.getRow(rowIndex));
  }


  @Override
  public Matrix log() {
    INDArray result = Transforms.log(matrix);
    return createNd4jMatrix(result);
  }

  @Override
  public Matrix logi() {
    this.matrix = Transforms.log(matrix);
    return this;
  }


  @Override
  public Matrix pow(int value) {
    INDArray result = Transforms.pow(matrix, value);
    return createNd4jMatrix(result);
  }

  @Override
  public Matrix powi(int value) {
    this.matrix = Transforms.pow(matrix, value);
    return this;
  }

  @Override
  public Matrix sigmoid() {
    INDArray result = Transforms.sigmoid(matrix);
    return createNd4jMatrix(result);
  }

  @Override
  public double[] toArray() {
    return matrix.transpose().data().asDouble();
  }
}
