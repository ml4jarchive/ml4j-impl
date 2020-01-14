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
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.floatmatrix.FloatMatrixFactory;

/**
 * Default JBlas MatrixFactory.
 * 
 * @author Michael Lavelle
 */
public class JBlasRowMajorMatrixFactory implements MatrixFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private FloatMatrixFactory floatMatrixFactory;

	public JBlasRowMajorMatrixFactory() {
		this.floatMatrixFactory = new DefaultFloatMatrixFactory();
	}

	@Override
	public Matrix createOnes(int rows, int columns) {
		return createJBlasMatrix(FloatMatrix.ones(columns, rows));
	}

	@Override
	public Matrix createOnes(int length) {
		return createJBlasMatrix(FloatMatrix.ones(length));
	}

	@Override
	public Matrix createMatrixFromRows(float[][] data) {
		float[][] translatedData = new float[data[0].length][data.length];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[0].length; j++) {
				translatedData[j][i] = data[i][j];
			}
		}
		return createJBlasMatrix(floatMatrixFactory.create(translatedData));
	}

	@Override
	public Matrix createMatrix() {
		return createJBlasMatrix(new FloatMatrix());
	}

	@Override
	public Matrix createMatrixFromColumnsByColumnsArray(int rows, int cols, float[] data) {

		float[] targetData = new float[rows * cols];
		for (int c = 0; c < cols; c++) {
			for (int r = 0; r < rows; r++) {
				int sourceDataIndex = c * cols + r;
				int targetDataIndex = r * rows + c;
				targetData[targetDataIndex] = data[sourceDataIndex];
			}
		}
		return createJBlasMatrix(new FloatMatrix(cols, rows, targetData));
	}

	@Override
	public Matrix createMatrix(int rows, int cols) {
		return createJBlasMatrix(floatMatrixFactory.create(cols, rows));
	}

	@Override
	public Matrix createZeros(int rows, int cols) {
		return createJBlasMatrix(floatMatrixFactory.create(cols, rows));
	}

	@Override
	public Matrix createRandn(int rows, int cols) {
		return createJBlasMatrix(FloatMatrix.randn(cols, rows));
	}

	@Override
	public Matrix createHorizontalConcatenation(Matrix first, Matrix second) {
		return first.appendHorizontally(second);
	}

	@Override
	public Matrix createRand(int rows, int cols) {
		return createJBlasMatrix(FloatMatrix.rand(cols, rows));
	}

	@Override
	public Matrix createVerticalConcatenation(Matrix first, Matrix second) {
		return first.appendVertically(second);
	}

	protected Matrix createJBlasMatrix(FloatMatrix matrix) {
		return new JBlasRowMajorMatrix(floatMatrixFactory, matrix, false);
	}

	protected Matrix createInterrimJBlasMatrix(FloatMatrix matrix) {
		return new JBlasRowMajorMatrix(floatMatrixFactory, matrix, false);
	}

	@Override
	public Matrix createMatrixFromRowsByRowsArray(int rows, int cols, float[] data) {
		return createJBlasMatrix(new FloatMatrix(cols, rows, data));
	}

	@Override
	public Matrix createMatrix(float[] data) {
		return createJBlasMatrix(new FloatMatrix(data));

	}

}
