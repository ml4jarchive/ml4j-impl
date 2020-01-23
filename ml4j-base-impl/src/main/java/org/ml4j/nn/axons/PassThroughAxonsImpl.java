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

import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

/**
 * Pass through (no-op) Axons implementation.
 * 
 * @author Michael Lavelle
 */
public class PassThroughAxonsImpl<N extends Neurons> implements Axons<N, N, PassThroughAxonsImpl<N>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private N leftNeurons;
	private N rightNeurons;

	/**
	 * @param leftNeurons  The left neurons.
	 * @param rightNeurons The right neurons.
	 */
	public PassThroughAxonsImpl(N leftNeurons, N rightNeurons) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;

		if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException("Left neuron and right neurons counts must be the same"
					+ leftNeurons.getNeuronCountIncludingBias() + ":" + rightNeurons.getNeuronCountIncludingBias());
		}
		if (leftNeurons.hasBiasUnit() != rightNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Left neuron and right neurons bias unit presence must be the same");
		}
		if (leftNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Left neurons with bias unit not supported");
		}
	}

	@Override
	public PassThroughAxonsImpl<N> dup() {
		return new PassThroughAxonsImpl<N>(leftNeurons, rightNeurons);
	}

	@Override
	public N getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public N getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public boolean isTrainable(AxonsContext context) {
		return false;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation input, AxonsActivation arg1, AxonsContext arg2) {
		return new AxonsActivationImpl(this, null, () -> input, input, leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation input, AxonsActivation arg1, AxonsContext arg2) {
		return new AxonsActivationImpl(this, null, () -> input, input, leftNeurons, rightNeurons);
	}
	
	

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return true;
	}
}
