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

package org.ml4j.nn.axons;

import org.ml4j.Matrix;

public class AxonsGradientImpl implements AxonsGradient {

	private TrainableAxons<?, ?, ?> axons;

	private Matrix weightsGradient;

	private Matrix leftToRightBiasGradient;

	private Matrix rightToLeftBiasGradient;

	/**
	 * @param axons
	 *            The TrainableAxons that generated this gradient.
	 * @param weightsGradient
	 *            The weights gradient.
	 * @param leftToRightBiasGradient
	 *            Left to right bias gradient.
	 */
	public AxonsGradientImpl(TrainableAxons<?, ?, ?> axons, Matrix weightsGradient, Matrix leftToRightBiasGradient) {
		super();
		this.axons = axons;
		this.weightsGradient = weightsGradient;
		this.leftToRightBiasGradient = leftToRightBiasGradient;
	}

	/**
	 * @param axons
	 *            The TrainableAxons that generated this gradient.
	 * @param weightsGradient
	 *            The weights gradient.
	 * @param leftToRightBiasGradient
	 *            Left to right bias gradient.
	 * @param rightToLeftBiasGradient
	 *            Right to left bias gradient.
	 */
	public AxonsGradientImpl(TrainableAxons<?, ?, ?> axons, Matrix weightsGradient, Matrix leftToRightBiasGradient,
			Matrix rightToLeftBiasGradient) {
		super();
		this.axons = axons;
		this.weightsGradient = weightsGradient;
		this.leftToRightBiasGradient = leftToRightBiasGradient;
		this.rightToLeftBiasGradient = rightToLeftBiasGradient;
	}

	public TrainableAxons<?, ?, ?> getAxons() {
		return axons;
	}

	@Override
	public Matrix getWeightsGradient() {
		return weightsGradient;
	}

	@Override
	public Matrix getLeftToRightBiasGradient() {
		return leftToRightBiasGradient;
	}

	@Override
	public Matrix getRightToLeftBiasGradient() {
		return rightToLeftBiasGradient;
	}

}
