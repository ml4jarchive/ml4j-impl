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

/**
 * Simple implementation of AxonsContext.
 * 
 * @author Michael Lavelle
 */
import org.ml4j.MatrixFactory;

public class AxonsContextImpl implements AxonsContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The MatrixFactory we configure for this context.
	 */
	private MatrixFactory matrixFactory;

	private boolean withFreezeOut;

	private float regularisationLambda;

	private float leftHandInputDropoutKeepProbability;

	private boolean trainingContext;

	/**
	 * Construct a new AxonsContext.
	 * 
	 * @param matrixFactory The MatrixFactory we configure for this context
	 * @param withFreezeOut Whether to freeze out these Axons.
	 */
	public AxonsContextImpl(MatrixFactory matrixFactory, boolean isTrainingContext, boolean withFreezeOut) {
		this.matrixFactory = matrixFactory;
		this.leftHandInputDropoutKeepProbability = 1f;
		this.withFreezeOut = withFreezeOut;
		this.trainingContext = isTrainingContext;
	}

	@Override
	public MatrixFactory getMatrixFactory() {
		return matrixFactory;
	}

	@Override
	public float getLeftHandInputDropoutKeepProbability() {
		return leftHandInputDropoutKeepProbability;
	}

	@Override
	public boolean isWithFreezeOut() {
		return withFreezeOut;
	}

	@Override
	public AxonsContext withFreezeOut(boolean withFreezeOut) {
		this.withFreezeOut = withFreezeOut;
		return this;
	}

	@Override
	public float getRegularisationLambda() {
		return regularisationLambda;
	}

	@Override
	public AxonsContext withLeftHandInputDropoutKeepProbability(float leftHandInputDropoutKeepProbability) {
		this.leftHandInputDropoutKeepProbability = leftHandInputDropoutKeepProbability;
		return this;
	}

	@Override
	public AxonsContext withRegularisationLambda(float regularisationLambda) {
		this.regularisationLambda = regularisationLambda;
		return this;
	}

	@Override
	public boolean isTrainingContext() {
		return trainingContext;
	}

	@Override
	public String toString() {
		return "AxonsContextImpl [matrixFactory=" + matrixFactory + ", withFreezeOut=" + withFreezeOut
				+ ", regularisationLambda=" + regularisationLambda + ", leftHandInputDropoutKeepProbability="
				+ leftHandInputDropoutKeepProbability + ", trainingContext=" + trainingContext + "]";
	}
	
	
}
