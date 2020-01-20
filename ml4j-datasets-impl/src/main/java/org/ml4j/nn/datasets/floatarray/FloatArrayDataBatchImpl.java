/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.datasets.floatarray;

import java.util.stream.Stream;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

import com.codepoetics.protonpack.StreamUtils;

public class FloatArrayDataBatchImpl extends DataBatchImpl<float[]> implements FloatArrayDataBatch {

	private int featureCount;
	private int batchSize;

	public FloatArrayDataBatchImpl(int featureCount, int batchSize) {
		super(batchSize);
		this.batchSize = batchSize;
		this.featureCount = featureCount;
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
	}

	public FloatArrayDataBatchImpl(Stream<float[]> data, int featureCount, int batchSize) {
		super(data, batchSize);
		this.featureCount = featureCount;
		this.batchSize = batchSize;
		if (batchSize == 0) {
			throw new IllegalArgumentException();
		}
		if (batchSize != size()) {
			throw new IllegalStateException();
		}
	}

	public Matrix getAsMatrix(MatrixFactory matrixFactory) {

		Matrix data = matrixFactory.createMatrix(batchSize, featureCount);

		StreamUtils.zipWithIndex(stream())
				.forEach(d -> addToMatrix(matrixFactory, data, d.getValue(), featureCount, (int) d.getIndex()));

		return data.transpose();
	}

	private void addToMatrix(MatrixFactory matrixFactory, Matrix matrix, float[] data, int featureCount, int row) {
		matrix.asEditableMatrix().putRow(row, matrixFactory.createMatrixFromRowsByRowsArray(1, featureCount, data));
	}

	@Override
	public NeuronsActivation toNeuronsActivation(MatrixFactory matrixFactory,
			NeuronsActivationFeatureOrientation featureOrientation) {

		Matrix dataMatrix = featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET
				? matrixFactory.createMatrix(featureCount, batchSize)
				: matrixFactory.createMatrix(batchSize, featureCount);

		Stream<float[]> floatStream = stream();
		if (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			StreamUtils.zipWithIndex(floatStream)
					.forEach(e -> dataMatrix.asEditableMatrix().putColumn((int) e.getIndex(),
							matrixFactory.createMatrixFromRowsByRowsArray(featureCount, 1, e.getValue())));
			return new NeuronsActivationImpl(new Neurons(dataMatrix.getRows(), false), dataMatrix, featureOrientation);

		} else {
			StreamUtils.zipWithIndex(floatStream).forEach(e -> dataMatrix.asEditableMatrix().putRow((int) e.getIndex(),
					matrixFactory.createMatrixFromRowsByRowsArray(featureCount, 1, e.getValue())));

			return new NeuronsActivationImpl(new Neurons(dataMatrix.getColumns(), false), dataMatrix,
					featureOrientation);

		}

	}

}
