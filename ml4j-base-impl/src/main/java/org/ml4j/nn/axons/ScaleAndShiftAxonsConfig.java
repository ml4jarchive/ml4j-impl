/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.axons;

import org.ml4j.Matrix;

/**
 * The configuration for ScaleAndShiftAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ScaleAndShiftAxonsConfig extends AxonsConfig {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Matrix initialScaleColumnVector;
	private Matrix initialShiftColumnVector;

	/**
	 * Configure scale and shift axons.
	 * 
	 * @param initialScaleRowVector
	 *            A row vector consisting of the scalings for each output neuron.
	 * @param initialShiftRowVector
	 *            A row vector consisting of the shifts for each output neuron.
	 */
	public ScaleAndShiftAxonsConfig(Matrix initialScaleColumnVector, Matrix initialShiftColumnVector) {
		super();
		this.initialScaleColumnVector = initialScaleColumnVector;
		this.initialShiftColumnVector = initialShiftColumnVector;
	}

	/**
	 * @return A column vector consisting of the scalings for each output neuron.
	 */
	public Matrix getScaleColumnVector() {
		return initialScaleColumnVector;
	}

	/**
	 * @return A column vector consisting of the shifts for each output neuron.
	 */
	public Matrix getShiftColumnVector() {
		return initialShiftColumnVector;
	}
}
