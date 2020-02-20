/*
 * Copyright 2020 the original author or authors.
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
package org.ml4j.nn.neurons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivationContext;

/**
 * Default implementation of NeuronsActivationContext
 * 
 * @author Michael Lavelle
 */
public class NeuronsActivationContextImpl implements NeuronsActivationContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private transient InheritableThreadLocal<MatrixFactory> matrixFactory;
	private transient InheritableThreadLocal<Boolean> isTrainingContext;

	public NeuronsActivationContextImpl(MatrixFactory matrixFactory, boolean isTrainingContext) {
		this.matrixFactory = new InheritableThreadLocal<>();
		this.matrixFactory.set(matrixFactory);
		this.isTrainingContext = new InheritableThreadLocal<>();
		this.isTrainingContext.set(isTrainingContext);
	}

	@Override
	public InheritableThreadLocal<MatrixFactory> getThreadLocalMatrixFactory() {
		return matrixFactory;
	}

	@Override
	public InheritableThreadLocal<Boolean> getThreadLocalIsTrainingContext() {
		return isTrainingContext;
	}

	@Override
	public void setMatrixFactory(MatrixFactory matrixFactory) {
		this.matrixFactory.set(matrixFactory);
	}

	@Override
	public void setTrainingContext(Boolean trainingContext) {
		this.isTrainingContext.set(trainingContext);
	}

	@Override
	public String toString() {
		return "NeuronsActivationContextImpl [isTrainingContext=" + isTrainingContext.get() + "]";
	}

	@Override
	public NeuronsActivationContext asNonTrainingContext() {
		return new NeuronsActivationContextImpl(matrixFactory.get(), false);
	}

	@Override
	public NeuronsActivationContext asTrainingContext() {
		return new NeuronsActivationContextImpl(matrixFactory.get(), true);
	}
	
	

}
