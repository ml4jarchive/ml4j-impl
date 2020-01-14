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
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
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
		return createNd4jMatrix(Nd4j.ones(rows, columns), false);
	}

	@Override
	public Matrix createOnes(int length) {
		return createNd4jMatrix(Nd4j.ones(length), false);
	}

	@Override
	public Matrix createMatrixFromRows(float[][] data) {
		// TODO - check immutable
		return createNd4jMatrix(
				Nd4j.create(new JBlasRowMajorMatrixFactory().createMatrixFromRows(data).getRowByRowArray(),
						new int[] { data.length, data[0].length }),
				false);
	}

	@Override
	public Matrix createMatrix() {
		return createNd4jMatrix(Nd4j.create(), false);
	}

	@Override
	public Matrix createMatrix(float[] data) {
		// TODO - check immutable
		return createNd4jMatrix(Nd4j.create(data), false);
	}

	@Override
	public Matrix createMatrixFromRowsByRowsArray(int rows, int cols, float[] data) {
		// TODO - check immutable
		return createNd4jMatrix(Nd4j.create(data, new int[] { rows, cols }), false);
	}

	@Override
	public Matrix createMatrix(int rows, int cols) {
		return createNd4jMatrix(Nd4j.create(new int[] { rows, cols }), false);
	}

	@Override
	public Matrix createZeros(int rows, int cols) {
		return createNd4jMatrix(Nd4j.create(new int[] { rows, cols }), false);
	}

	@Override
	public Matrix createRandn(int rows, int cols) {
		return createNd4jMatrix(Nd4j.randn(rows, cols), false);
	}

	@Override
	public Matrix createHorizontalConcatenation(Matrix first, Matrix second) {
		return first.appendHorizontally(second);
	}

	@Override
	public Matrix createRand(int rows, int cols) {
		return createNd4jMatrix(Nd4j.rand(rows, cols), false);
	}

	@Override
	public Matrix createVerticalConcatenation(Matrix first, Matrix second) {
		return first.appendVertically(second);
	}

	protected Nd4jMatrix createNd4jMatrix(INDArray matrix, boolean immutable) {
		return new Nd4jMatrix(matrix, immutable);
	}

	@Override
	public Matrix createMatrixFromColumnsByColumnsArray(int rows, int cols, float[] data) {
		throw new UnsupportedOperationException("Not implemented yet");
	}
}
