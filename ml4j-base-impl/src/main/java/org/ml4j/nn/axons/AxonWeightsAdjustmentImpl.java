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
package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.Matrix;

/**
 * Default implementation of AxonWeightsAdjustment.
 * 
 * @author Michael Lavelle
 *
 */
public class AxonWeightsAdjustmentImpl implements AxonWeightsAdjustment {

	private Matrix connectionWeights;
	private Matrix leftToRightBiases;
	private Matrix rightToLeftBiases;

	public AxonWeightsAdjustmentImpl(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
	}

	public AxonWeightsAdjustmentImpl(Matrix connectionWeights, Matrix leftToRightBiases) {
		this.connectionWeights = connectionWeights;
		this.leftToRightBiases = leftToRightBiases;
	}

	public AxonWeightsAdjustmentImpl(Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		this.connectionWeights = connectionWeights;
		this.leftToRightBiases = leftToRightBiases;
		this.rightToLeftBiases = rightToLeftBiases;
	}

	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public Optional<Matrix> getLeftToRightBiases() {
		return Optional.ofNullable(leftToRightBiases);
	}

	@Override
	public Optional<Matrix> getRightToLeftBiases() {
		return Optional.ofNullable(rightToLeftBiases);
	}
}
