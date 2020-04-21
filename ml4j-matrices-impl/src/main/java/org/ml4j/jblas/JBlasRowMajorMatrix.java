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

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.floatarray.FloatArrayFactory;
import org.ml4j.floatmatrix.FloatMatrixFactory;

/**
 * Default JBlas matrix implementation.
 * 
 * @author Michael Lavelle
 */
public class JBlasRowMajorMatrix implements Matrix, EditableMatrix, InterrimMatrix {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	protected FloatMatrix matrix;
	protected FloatMatrixFactory floatMatrixFactory;
	protected FloatArrayFactory floatArrayFactory;
	protected boolean immutable;
	protected JBlasRowMajorMatrixFactory jblasRowMajorMatrixFactory;

	public JBlasRowMajorMatrix(JBlasRowMajorMatrixFactory jblasRowMajorMatrixFactory, FloatMatrixFactory floatMatrixFactory, FloatArrayFactory floatArrayFactory, 
			FloatMatrix matrix, boolean immutable) {
		this.matrix = matrix;
		this.floatMatrixFactory = floatMatrixFactory;
		this.immutable = immutable;
		this.floatArrayFactory = floatArrayFactory;
		this.jblasRowMajorMatrixFactory = jblasRowMajorMatrixFactory;
	}

	private FloatMatrix getFloatMatrix() {
		return getMatrix();
	}

	/**
	 * Create a new JBlas FloatMatrix from the Matrix.
	 * 
	 * @param matrix The matrix we want to convert to a FloatMatrix.
	 * @return The resulting FloatMatrix.
	 */
	protected FloatMatrix createJBlasFloatMatrix(Matrix matrix) {
		if (matrix instanceof JBlasRowMajorMatrix) {
			return ((JBlasRowMajorMatrix) matrix).getFloatMatrix();
		} else {
			throw new UnsupportedOperationException();
		}
	}

	protected Matrix createJBlasMatrix(FloatMatrix matrix, boolean immutable) {
		return jblasRowMajorMatrixFactory.createJBlasMatrix(matrix, immutable);
	}

	@Override
	public Matrix add(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(getMatrix().add(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix addColumnVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(getMatrix().addRowVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix addRowVector(Matrix other) {
		if (other.getColumns() != this.getColumns()) {
			throw new IllegalArgumentException("Columns do not match");
		}
		return createJBlasMatrix(getMatrix().addColumnVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix add(float value) {
		return createJBlasMatrix(getMatrix().addi(value, floatMatrixFactory.create(getColumns(), getRows())), false);
	}

	@Override
	public Matrix sub(float value) {
		return createJBlasMatrix(getMatrix().subi(value, floatMatrixFactory.create(getColumns(), getRows())), false);
	}

	@Override
	public EditableMatrix addi(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		matrix.addi(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix addiColumnVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		getMatrix().addiRowVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix addiRowVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		getMatrix().addiColumnVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix addi(float value) {
		getMatrix().addi(value);
		return this;
	}

	@Override
	public EditableMatrix subi(float value) {
		getMatrix().subi(value);
		return this;
	}

	@Override
	public int argmax() {
		return getMatrix().argmax();
	}

	public Matrix copy(Matrix other) {
		return createJBlasMatrix(getMatrix().copy(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix div(float value) {
		return createJBlasMatrix(getMatrix().divi(value, floatMatrixFactory.create(getColumns(), getRows())), false);
	}

	@Override
	public Matrix div(Matrix other) {
		return createJBlasMatrix(
				getMatrix().divi(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())),
				false);
	}

	@Override
	public EditableMatrix divi(float value) {
		getMatrix().divi(value);
		return this;
	}

	@Override
	public EditableMatrix divi(Matrix other) {
		getMatrix().divi(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix diviColumnVector(Matrix other) {
		getMatrix().diviRowVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix diviRowVector(Matrix other) {
		getMatrix().diviColumnVector(createJBlasFloatMatrix(other));
		return this;
	}

	public float dot(Matrix other) {
		return getMatrix().dot(createJBlasFloatMatrix(other));
	}

	public int[] findIndices() {
		return getMatrix().findIndices();
	}

	public float get(int index) {
		return getMatrix().get(index);
	}

	@Override
	public float get(int row, int col) {
		return getMatrix().get(col, row);
	}

	public Matrix get(int[] rows, int[] cols) {
		return createJBlasMatrix(getMatrix().get(cols, rows), false);
	}

	@Override
	public int getColumns() {
		return getMatrix().getRows();
	}

	public Matrix getColumns(int[] cols) {
		return createJBlasMatrix(getMatrix().getRows(cols), false);
	}

	@Override
	public int getLength() {
		return getMatrix().getLength();
	}

	public Matrix getRowRange(int avalue, int bvalue, int cvalue) {
		return createJBlasMatrix(getMatrix().getColumnRange(avalue, bvalue, cvalue), false);
	}

	@Override
	public int getRows() {
		return getMatrix().getColumns();
	}

	@Override
	public Matrix getRows(int[] rows) {
		return createJBlasMatrix(getMatrix().getColumns(rows), false);
	}

	public Matrix mmul(Matrix other, Matrix target) {
		return createJBlasMatrix(getMatrix().mmuli(createJBlasFloatMatrix(other), createJBlasFloatMatrix(target)),
				false);
	}

	@Override
	public Matrix mmul(Matrix other)  {
		
		FloatMatrix o = createJBlasFloatMatrix(other);
		FloatMatrix t = getMatrix();

		return createJBlasMatrix(o.mmuli(t, floatMatrixFactory.create(other.getColumns(), getRows())), false);

	}
	

	@Override
	public Matrix mul(float value) {
		return createJBlasMatrix(getMatrix().mul(value), false);
	}

	@Override
	public Matrix mul(Matrix other) {
		return createJBlasMatrix(
				getMatrix().muli(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())),
				false);
	}

	@Override
	public Matrix mulColumnVector(Matrix other) {
		return createJBlasMatrix(getMatrix().mulRowVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix mulRowVector(Matrix other) {
		return createJBlasMatrix(getMatrix().mulColumnVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix divRowVector(Matrix other) {
		return createJBlasMatrix(getMatrix().divColumnVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix divColumnVector(Matrix other) {
		return createJBlasMatrix(getMatrix().divRowVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public EditableMatrix muli(Matrix other) {
		getMatrix().muli(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix muliRowVector(Matrix other) {
		getMatrix().muliColumnVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix muliColumnVector(Matrix other) {
		getMatrix().muliRowVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix muli(float value) {
		getMatrix().muli(value);
		return this;
	}

	@Override
	public void put(int index, float value) {
		getMatrix().put(index, value);
	}

	@Override
	public void put(int row, int col, float value) {
		getMatrix().put(col, row, value);
	}

	public void put(int[] indices, int cvalue, Matrix other) {
		getMatrix().put(indices, cvalue, createJBlasFloatMatrix(other));
	}

	@Override
	public void putColumn(int columnIndex, Matrix other) {
		getMatrix().putRow(columnIndex, createJBlasFloatMatrix(other));
	}

	@Override
	public void putRow(int rowIndex, Matrix other) {
		getMatrix().putColumn(rowIndex, createJBlasFloatMatrix(other));
	}

	public void reshape(int newRows, int newColumns) {
		getMatrix().reshape(newColumns, newRows);
	}

	public int[] rowArgmaxs() {
		return getMatrix().columnArgmaxs();
	}

	@Override
	public int[] columnArgmaxs() {
		return getMatrix().rowArgmaxs();
	}

	@Override
	public Matrix rowSums() {
		return createJBlasMatrix(getMatrix().columnSums(), false);
	}

	@Override
	public Matrix columnSums() {
		return createJBlasMatrix(getMatrix().rowSums(), false);
	}

	@Override
	public Matrix sub(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(
				getMatrix().subi(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())),
				false);

	}

	@Override
	public Matrix subColumnVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(matrix.subRowVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public Matrix subRowVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(getMatrix().subColumnVector(createJBlasFloatMatrix(other)), false);
	}

	@Override
	public EditableMatrix subi(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		getMatrix().subi(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix subiColumnVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		getMatrix().subiRowVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public EditableMatrix subiRowVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		matrix.subiColumnVector(createJBlasFloatMatrix(other));
		return this;
	}

	@Override
	public float sum() {
		return getMatrix().sum();
	}

	public float[][] toArray2() {
		float[][] result = floatArrayFactory.createFloatArray(getRows(), getColumns());
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getColumns(); c++) {
				result[r][c] = get(r, c);
			}
		}
		return result;
	}

	@Override
	public Matrix transpose() {
		return createJBlasMatrix(getMatrix().transpose(), false);
	}

	@Override
	public Matrix appendHorizontally(Matrix other) {
		FloatMatrix result = FloatMatrix.concatVertically(getMatrix(), createJBlasFloatMatrix(other));
		return createJBlasMatrix(result, false);
	}

	@Override
	public Matrix appendVertically(Matrix other) {
		FloatMatrix result = FloatMatrix.concatHorizontally(getMatrix(), createJBlasFloatMatrix(other));
		return createJBlasMatrix(result, false);
	}

	public Matrix asCudaMatrix() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	public Matrix asJBlasMatrix() {
		return this;
	}

	@Override
	public Matrix dup() {
		return createJBlasMatrix(getMatrix().dup(), false);
	}

	@Override
	public EditableMatrix expi() {
		this.matrix = MatrixFunctions.expi(getMatrix());
		return this;
	}

	@Override
	public Matrix getColumn(int columnIndex) {
		return createJBlasMatrix(getMatrix().getRow(columnIndex), false);
	}

	public Matrix getRow(int rowIndex) {
		return createJBlasMatrix(getMatrix().getColumn(rowIndex), false);
	}

	public Matrix log() {
		FloatMatrix result = MatrixFunctions.log(getMatrix());
		return createJBlasMatrix(result, false);
	}

	public Matrix logi() {
		this.matrix = MatrixFunctions.logi(getMatrix());
		return this;
	}

	public Matrix pow(int value) {
		FloatMatrix result = MatrixFunctions.pow(getMatrix(), value);
		return createJBlasMatrix(result, false);
	}

	public Matrix powi(int value) {
		this.matrix = MatrixFunctions.powi(getMatrix(), value);
		return this;
	}

	@Override
	public Matrix sigmoid() {
		FloatMatrix result = MatrixFunctions
				.expi(getMatrix().muli(-1, floatMatrixFactory.create(getColumns(), getRows())));
		result.addi(1);
		MatrixFunctions.powi(result, -1);
		return createJBlasMatrix(result, false);
	}

	public float[] toColumnByColumnArray() {
		float[] result = floatArrayFactory.createFloatArray(getRows() * getColumns());
		int index = 0;
		for (int c = 0; c < getColumns(); c++) {
			for (int r = 0; r < getRows(); r++) {
				result[index] = get(r, c);
				index++;
			}
		}
		return result;
	}

	@Override
	public void close() {
		if (this.matrix != null) {
			this.matrix = null;
		}
	}

	@Override
	public InterrimMatrix asInterrimMatrix() {
		return this;
	}

	// @Override
	public float[] toRowByRowArray() {
		return getMatrix().toArray();
	}

	private FloatMatrix getMatrix() {
		if (matrix == null) {
			throw new IllegalStateException("Matrix has been clased");
		}

		return matrix;
	}

	@Override
	public float[] getRowByRowArray() {
		return getMatrix().data;
	}

	public float[] getColumnByColumnArray() {
		return toColumnByColumnArray();
	}

	public float[] getData() {
		return getMatrix().data;
	}

	@Override
	public EditableMatrix asEditableMatrix() {
		if (isImmutable()) {
			throw new IllegalStateException("Matrix is immutable");
		}

		return this;
	}

	@Override
	public boolean isImmutable() {
		return immutable;
	}

	@Override
	public void setImmutable(boolean immutable) {
		this.immutable = immutable;
	}

	@Override
	public Matrix softDup() {
		return createJBlasMatrix(softDupFloatMatrix(getMatrix()), false);
	}

	public FloatMatrix softDupFloatMatrix(FloatMatrix matrix) {
		return new FloatMatrix(matrix.getRows(), matrix.getColumns(), matrix.data);
	}

	@Override
	public boolean isClosed() {
		return matrix == null;
	}
}
