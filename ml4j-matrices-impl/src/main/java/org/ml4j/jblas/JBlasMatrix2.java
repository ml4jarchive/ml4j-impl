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

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;


/**
 * Default JBlas matrix implementation.
 * 
 * @author Michael Lavelle
 */
public class JBlasMatrix2 implements Matrix, EditableMatrix, InterrimMatrix {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public FloatMatrix matrix;

	private String stackTrace;
	private String closeStackTrace;
	private FloatMatrixFactory floatMatrixFactory;
	private UUID id;
	private boolean persisted;
	private short[] compressed;
	private int compressedRows;
	private int compressedColumns;
	private boolean immutable;

	private static Set<String> stackTraces = new HashSet<>();

	public JBlasMatrix2(FloatMatrixFactory floatMatrixFactory, FloatMatrix matrix, boolean immutable) {
		// this.matrix = matrix;
		this.matrix = matrix;
		this.floatMatrixFactory = floatMatrixFactory;
		this.immutable = immutable;
		/*
		 * ByteArrayOutputStream os = new ByteArrayOutputStream(); PrintWriter s = new
		 * PrintWriter(os); new RuntimeException().printStackTrace(s); s.flush();
		 * s.close(); stackTrace = os.toString(); this.id = UUID.randomUUID();
		 */
	}

	private FloatMatrix getFloatMatrix() {
		return getMatrix();
	}

	/**
	 * Create a new JBlas FloatMatrix from the Matrix.
	 * 
	 * @param matrix
	 *            The matrix we want to convert to a FloatMatrix.
	 * @return The resulting FloatMatrix.
	 */
	private FloatMatrix createJBlasFloatMatrix(Matrix matrix) {
		if (matrix instanceof JBlasMatrix2) {
			return ((JBlasMatrix2) matrix).getFloatMatrix();
		} else if (false) {// matrix instanceof Nd4jMatrix) {
			throw new UnsupportedOperationException();

			//return floatMatrixFactory.create(getRows(), getColumns(), ((Nd4jMatrix)matrix).getData());
			//System.out.println(matrix.getClass().getName());
			//System.out.println(this.stackTrace);
			// return new FloatMatrix(matrix.toArray2());
		} else {
			throw new UnsupportedOperationException();
		}
	}

	protected Matrix createJBlasMatrix(FloatMatrix matrix, boolean immutable) {
		return new JBlasMatrix2(floatMatrixFactory, matrix, immutable);
	}

	// Unused
	@Override
	public Matrix add(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(getMatrix().add(createJBlasFloatMatrix(other)), false);
	}

	// Unused
	@Override
	public Matrix addColumnVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
		}
		return createJBlasMatrix(getMatrix().addRowVector(createJBlasFloatMatrix(other)), false);
	}

	// Unused
	@Override
	public Matrix addRowVector(Matrix other) {
		if (other.getRows() != this.getRows()) {
			throw new IllegalArgumentException("Rows do not match");
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

	//@Override
	public int argmax() {
		return getMatrix().argmax();
	}

	//@Override
	public Matrix copy(Matrix other) {
		return createJBlasMatrix(getMatrix().copy(createJBlasFloatMatrix(other)), false);
	}

	// Creates new average gradient matrices from totals
	@Override
	public Matrix div(float value) {
		return createJBlasMatrix(getMatrix().divi(value, floatMatrixFactory.create(getColumns(), getRows())), false);
	}

	// Unused
	@Override
	public Matrix div(Matrix other) {
		return createJBlasMatrix(
				getMatrix().divi(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())), false);
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

	//@Override
	public float dot(Matrix other) {
		return getMatrix().dot(createJBlasFloatMatrix(other));
	}

	//@Override
	public int[] findIndices() {
		return getMatrix().findIndices();
	}

	// unused
	//@Override
	public float get(int index) {
		return getMatrix().get(index);
	}

	// Convolutional converter - temp
	// Pooling formatter - to get values that arent' -1
	// Convolutional converter - temp
	// Max pool axons, to get value from a max int index
	// Batch norm to get variance
	// Multi class cost function - limitlog
	// ReluActivationFunction - to get values <=0
	@Override
	public float get(int row, int col) {
		return getMatrix().get(col, row);
	}

	// Only rbm
	//@Override
	public Matrix get(int[] rows, int[] cols) {
		return createJBlasMatrix(getMatrix().get(cols, rows), false);
	}

	@Override
	public int getColumns() {
		return getMatrix().getRows();
	}

	// Only rbm and getting data batch
	//@Override
	public Matrix getColumns(int[] cols) {
		return createJBlasMatrix(getMatrix().getRows(cols), false);
	}

	@Override
	public int getLength() {
		return getMatrix().getLength();
	}

	//@Override
	public Matrix getRowRange(int avalue, int bvalue, int cvalue) {
		return createJBlasMatrix(getMatrix().getColumnRange(avalue, bvalue, cvalue), false);
	}

	@Override
	public int getRows() {
		return getMatrix().getColumns();
	}

	// only rbm and creating supervised networks
	@Override
	public Matrix getRows(int[] rows) {
		return createJBlasMatrix(getMatrix().getColumns(rows), false);
	}

	// TODO
	//@Override
	public Matrix mmul(Matrix other, Matrix target) {
		return createJBlasMatrix(getMatrix().mmuli(createJBlasFloatMatrix(other), createJBlasFloatMatrix(target)), false);
	}

	// TODO
	@Override
	public Matrix mmul(Matrix other) {
		if (false) {
			return createJBlasMatrix(floatMatrixFactory.create(other.getColumns(), getRows()), false);
		}
		FloatMatrix o = createJBlasFloatMatrix(other);
		FloatMatrix t = getMatrix();

		return createJBlasMatrix(o.mmuli(t,
				floatMatrixFactory.create(other.getColumns(), getRows())), false);
	}

	@Override
	public Matrix mul(float value) {
		return createJBlasMatrix(getMatrix().mul(value), false);
	}

	// TODO
	@Override
	public Matrix mul(Matrix other) {
		// Check this
		return createJBlasMatrix(
				getMatrix().muli(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())), false);
	}

	@Override
	public Matrix mulColumnVector(Matrix other) {
		// Check this
		return createJBlasMatrix(getMatrix().mulRowVector(createJBlasFloatMatrix(other)), false);
	}

	// Unused
	@Override
	public Matrix mulRowVector(Matrix other) {
		// Check this
		return createJBlasMatrix(getMatrix().mulColumnVector(createJBlasFloatMatrix(other)), false);
	}
	
	@Override
	public Matrix divRowVector(Matrix other) {
		// Check this
		return createJBlasMatrix(getMatrix().divColumnVector(createJBlasFloatMatrix(other)), false);
	}
	
	@Override
	public Matrix divColumnVector(Matrix other) {
		// Check this
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

	// Unused
	@Override
	public void put(int index, float value) {
		getMatrix().put(index, value);
	}

	// Same as get
	@Override
	public void put(int row, int col, float value) {
		getMatrix().put(col, row, value);
	}

	//@Override
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

	//@Override
	public void reshape(int newRows, int newColumns) {
		getMatrix().reshape(newColumns, newRows);
	}

	//@Override
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
				getMatrix().subi(createJBlasFloatMatrix(other), floatMatrixFactory.create(getColumns(), getRows())), false);

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

	//@Override
	public float[][] toArray2() {
		float[][] result = new float[getRows()][getColumns()];
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getColumns(); c++) {
				result[r][c] = get(r, c);
			}
		}
		return result;
	}

	@Override
	public Matrix transpose() {
		// TODO ML Need to check this
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

	//@Override
	public Matrix asCudaMatrix() {
		System.out.println("Converting to Cuda Matrix");
		// new RuntimeException().printStackTrace();
		return null;
		//return new Nd4jMatrixFactory().createMatrixFromRowsByRowsArray(getRows(), getColumns(), matrix.data);
	}

	//@Override
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

	//@Override
	public Matrix getRow(int rowIndex) {
		return createJBlasMatrix(getMatrix().getColumn(rowIndex), false);
	}

	// No logi with target matrix available
	//s@Override
	public Matrix log() {
		FloatMatrix result = MatrixFunctions.log(getMatrix());
		return createJBlasMatrix(result, false);
	}

	//@Override
	public Matrix logi() {
		this.matrix = MatrixFunctions.logi(getMatrix());
		return this;
	}

	// Unused
	//@Override
	public Matrix pow(int value) {
		FloatMatrix result = MatrixFunctions.pow(getMatrix(), value);
		return createJBlasMatrix(result, false);
	}

	//@Override
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

	//@Override
	public float[] toColumnByColumnArray() {
		float[] result = new float[getRows() * getColumns()];
		int index = 0;
		for (int c = 0; c < getColumns(); c++) {
			for (int r = 0; r < getRows(); r++) {
				result[index] = get(r, c);
				index++;
			}
		}
		return result;

		/*
		 * double[] result = new double[getRows() * getColumns()]; double[][] rowsArray
		 * = matrix.toArray2(); int index = 0; for (double[] row : rowsArray) {
		 * System.arraycopy(row, 0, result, index * getColumns(), getColumns());
		 * index++; } return result;
		 */
	}

	@Override
	protected void finalize() throws Throwable {
		if (matrix != null) {
			// matrixCount.decrementAndGet();
			// addMatrixToCache(matrix);
			if (!stackTraces.contains(this.stackTrace)) {
				// System.out.println(this.stackTrace);
				stackTraces.add(stackTrace);
			}
		}
	}

	@Override
	public void close() {
		if (this.matrix != null) {
			this.matrix = null;
			// ]addMatrixToCache(this.matrix);
			// matrixCount.decrementAndGet();
			// matrix.data = null;
			
			 ByteArrayOutputStream os = new ByteArrayOutputStream(); PrintWriter s = new
			PrintWriter(os); new RuntimeException().printStackTrace(s); s.flush();
			 s.close(); 
			 closeStackTrace = os.toString();
			

		}
	}

	@Override
	public InterrimMatrix asInterrimMatrix() {
		return this;
	}

	//@Override
	public float[] toRowByRowArray() {
		return getMatrix().toArray();
	}

	private FloatMatrix getMatrix() {
		if (matrix == null) {
			System.out.println(closeStackTrace);
			throw new IllegalStateException("Matrix has been clased");
		}

		return matrix;
	}

	@Override
	public float[] getRowByRowArray() {
		return getMatrix().data;
	}

	//@Override
	public float[] getColumnByColumnArray() {
		return toColumnByColumnArray();
	}

	//@Override
	public float[] getData() {
		return getMatrix().data;
	}
	
	@Override
	public EditableMatrix asEditableMatrix() {
		if (immutable) {
			throw new IllegalStateException();
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
}
