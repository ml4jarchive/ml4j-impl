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

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.jblas.JBlasRowMajorMatrix;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Default Nd4j matrix implementation.
 * 
 * @author Michael Lavelle
 */
public class Nd4jRowMajorMatrix implements Matrix, EditableMatrix, InterrimMatrix {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public INDArray matrix;
	private boolean immutable;

	public Nd4jRowMajorMatrix(INDArray matrix, boolean immutable) {
		this.matrix = matrix;
		this.immutable = immutable;
	}

	private INDArray getIndArray() {
		if (matrix == null) {
			throw new IllegalStateException("Matrix has been closed");
		}
		return matrix;
	}

	/**
	 * Create a new n4j4j INDArray from the Matrix.
	 * 
	 * @param matrix The matrix we want to convert to a INDArray.
	 * @return The resulting INDArray.
	 */
	private INDArray getNd4jIndArray(Matrix matrix) {
		if (matrix instanceof Nd4jRowMajorMatrix) {
			return ((Nd4jRowMajorMatrix) matrix).getIndArray();
		} else if (matrix instanceof JBlasRowMajorMatrix) {
			return Nd4j.create(matrix.dup().getRowByRowArray(), new int[] { matrix.getRows(), matrix.getColumns() });
		} else {
			throw new UnsupportedOperationException("Only Nd4j and JBlas matrices currently supported");
		}
	}

	protected Nd4jRowMajorMatrix createNd4jMatrix(INDArray matrix, boolean immutable) {
		return new Nd4jRowMajorMatrix(matrix, immutable);
	}

	@Override
	public Matrix add(Matrix other) {
		return createNd4jMatrix(matrix.add(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix add(float value) {
		return createNd4jMatrix(matrix.add(value), immutable);
	}

	@Override
	public EditableMatrix addi(Matrix other) {
		this.matrix = matrix.addi(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix addi(float value) {
		this.matrix = this.matrix.addi(value);
		return this;
	}

	@Override
	public Matrix div(float value) {
		return createNd4jMatrix(matrix.div(value), immutable);
	}

	@Override
	public Matrix div(Matrix other) {
		return createNd4jMatrix(matrix.div(getNd4jIndArray(other)), immutable);
	}

	@Override
	public EditableMatrix divi(float value) {
		this.matrix = matrix.divi(value);
		return this;
	}

	@Override
	public EditableMatrix divi(Matrix other) {
		this.matrix = matrix.divi(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix diviColumnVector(Matrix other) {
		this.matrix = matrix.diviColumnVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public float get(int index) {
		return matrix.getFloat(index);
	}

	@Override
	public float get(int row, int col) {
		return matrix.getFloat(row, col);
	}

	@Override
	public int getColumns() {
		return matrix.columns();
	}

	@Override
	public int getLength() {
		return matrix.length();
	}

	@Override
	public int getRows() {
		return matrix.rows();
	}

	@Override
	public Matrix mul(float value) {
		return createNd4jMatrix(matrix.mul(value), false);
	}

	@Override
	public Matrix mul(Matrix other) {
		return createNd4jMatrix(matrix.mul(getNd4jIndArray(other)), false);
	}

	@Override
	public EditableMatrix muli(Matrix other) {

		this.matrix = matrix.muli(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix muli(float value) {
		this.matrix = matrix.muli(value);
		return this;
	}

	@Override
	public void put(int index, float value) {
		matrix.putScalar(index, value);
	}

	@Override
	public void put(int row, int col, float value) {
		matrix.putScalar(row, col, value);
	}

	@Override
	public void putColumn(int columnIndex, Matrix other) {
		matrix.putColumn(columnIndex, getNd4jIndArray(other));
	}

	@Override
	public void putRow(int rowIndex, Matrix other) {
		matrix.putRow(rowIndex, getNd4jIndArray(other));
	}

	@Override
	public void reshape(int newRows, int newColumns) {
		this.matrix = matrix.reshape(newRows, newColumns);
	}

	@Override
	public Matrix sub(Matrix other) {
		return createNd4jMatrix(matrix.sub(getNd4jIndArray(other)), false);
	}

	@Override
	public EditableMatrix subi(Matrix other) {
		this.matrix = matrix.subi(getNd4jIndArray(other));
		return this;
	}

	@Override
	public Matrix appendHorizontally(Matrix other) {
		return createNd4jMatrix(Nd4j.hstack(matrix, getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix appendVertically(Matrix other) {
		return createNd4jMatrix(Nd4j.vstack(matrix, getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix dup() {
		return createNd4jMatrix(matrix.dup(), false);
	}

	@Override
	public EditableMatrix expi() {
		this.matrix = Transforms.exp(matrix);
		return this;
	}

	@Override
	public Matrix log() {
		INDArray result = Transforms.log(matrix);
		return createNd4jMatrix(result, false);
	}

	@Override
	public Matrix logi() {
		this.matrix = Transforms.log(matrix);
		return this;
	}

	@Override
	public Matrix sigmoid() {
		INDArray result = Transforms.sigmoid(matrix);
		return createNd4jMatrix(result, false);
	}

	@Override
	public Matrix sub(float v) {
		return createNd4jMatrix(matrix.sub(v), false);
	}

	@Override
	public void close() {
		if (this.matrix != null) {
			this.matrix = null;
		}
	}

	@Override
	public boolean isClosed() {
		return matrix == null;
	}

	@Override
	public EditableMatrix asEditableMatrix() {
		if (immutable) {
			throw new IllegalStateException("Matrix is immutable");
		}
		return this;
	}

	@Override
	public InterrimMatrix asInterrimMatrix() {
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
	public EditableMatrix subi(float v) {
		this.matrix = matrix.subi(v);
		return this;
	}

	@Override
	public EditableMatrix subiColumnVector(Matrix other) {
		this.matrix = matrix.subiColumnVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix subiRowVector(Matrix other) {
		this.matrix = matrix.subiRowVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix diviRowVector(Matrix other) {
		this.matrix = matrix.diviRowVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix addiRowVector(Matrix other) {
		this.matrix = matrix.addiRowVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix addiColumnVector(Matrix other) {
		this.matrix = matrix.addiColumnVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix muliColumnVector(Matrix other) {
		this.matrix = matrix.muliColumnVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public EditableMatrix muliRowVector(Matrix other) {
		this.matrix = matrix.muliRowVector(getNd4jIndArray(other));
		return this;
	}

	@Override
	public float[] getRowByRowArray() {
		return (float[]) matrix.data().asFloat();
	}

	@Override
	public float[] toColumnByColumnArray() {
		return asJBlasMatrix().getColumnByColumnArray();
	}

	@Override
	public Matrix columnSums() {
		return asJBlasMatrix().columnSums();
	}

	@Override
	public int[] columnArgmaxs() {
		return asJBlasMatrix().columnArgmaxs();
	}

	@Override
	public Matrix mulColumnVector(Matrix other) {
		return createNd4jMatrix(matrix.mulColumnVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix mulRowVector(Matrix other) {
		return createNd4jMatrix(matrix.mulRowVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix addColumnVector(Matrix other) {
		return createNd4jMatrix(matrix.addColumnVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix addRowVector(Matrix other) {
		return createNd4jMatrix(matrix.addRowVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix divColumnVector(Matrix other) {
		return createNd4jMatrix(matrix.divColumnVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix divRowVector(Matrix other) {
		return createNd4jMatrix(matrix.divRowVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix subColumnVector(Matrix other) {
		return createNd4jMatrix(matrix.subColumnVector(getNd4jIndArray(other)), false);
	}

	@Override
	public Matrix subRowVector(Matrix other) {
		return createNd4jMatrix(matrix.subRowVector(getNd4jIndArray(other)), false);
	}

	
	// The below methods are implemented as they are because obvious implementations are not available in Nd4j, or there are issues using the
	// Nd4j versions.  TODO - Migrate the below methods to Nd4j

	@Override
	public Matrix mmul(Matrix other) {
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix().mmul(asJBlasMatrix(other))), false);
		// return createNd4jMatrix(matrix.mmul(createNd4jIndArray(other)));
	}

	@Override
	public Matrix get(int[] rows, int[] cols) {
		// throw new UnsupportedOperationException("Not yet implemented");
		// TODO check immutable
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix().get(rows, cols)), immutable);
	}

	@Override
	public int argmax() {
		return asJBlasMatrix().argmax();
	}

	@Override
	public Matrix getRows(int[] rows) {
		// TODO - check immutable
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix()).getRows(rows), immutable);
	}

	@Override
	public Matrix getColumns(int[] cols) {
		// TODO - check immutable
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix()).getColumns(cols), immutable);
	}

	@Override
	public Matrix rowSums() {
		return asJBlasMatrix().rowSums();
	}

	@Override
	public Matrix transpose() {
		// return createNd4jMatrix(matrix.transpose());
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix().transpose()), false);
	}

	@Override
	public float sum() {
		return asJBlasMatrix().sum();
	}

	@Override
	public Matrix getColumn(int columnIndex) {
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix().getColumn(columnIndex)), false);
		// return createNd4jMatrix(matrix.getColumn(columnIndex));
	}

	@Override
	public Matrix getRow(int rowIndex) {
		return createNd4jMatrix(getNd4jIndArray(asJBlasMatrix().getRow(rowIndex)), false);
	}

	@Override
	public float[] getColumnByColumnArray() {
		return asJBlasMatrix().getColumnByColumnArray();
	}

	// The below methods are utility methods to aid the temporary JBlas matrix implementations as described above.
	
	private Matrix asJBlasMatrix() {
		return asJBlasMatrix(this);
	}

	private Matrix asJBlasMatrix(Matrix other) {
		if (other instanceof JBlasRowMajorMatrix) {
			return other;
		} else {
			return
			// TODO- check immutable
			new JBlasRowMajorMatrixFactory().createMatrixFromRowsByRowsArray(other.getRows(), other.getColumns(),
					other.getRowByRowArray());
		}
	}
	
	// The below methods are not yet implemented - TODO
	
	@Override
	public Matrix softDup() {
		// TODO
		throw new UnsupportedOperationException("Not implemented yet");
		// return createNd4jMatrix(softDupIndArray(matrix), immutable);
	}

	public Nd4jRowMajorMatrix softDupIndArray(INDArray matrix) {
		// TODO
		return createNd4jMatrix(matrix, immutable);
	}
	
}
