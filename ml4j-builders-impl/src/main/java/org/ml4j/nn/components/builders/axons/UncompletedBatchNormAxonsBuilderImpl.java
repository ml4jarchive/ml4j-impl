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
package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.BatchNormConfig.BatchNormDimension;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.neurons.Neurons;

public class UncompletedBatchNormAxonsBuilderImpl<C extends AxonsBuilder<?>>
		extends UncompletedAxonsBuilderImpl<Neurons, C> implements UncompletedBatchNormAxonsBuilder<Neurons, C> {

	private WeightsMatrix gamma;
	private BiasMatrix beta;
	private Matrix mean;
	private Matrix variance;

	public UncompletedBatchNormAxonsBuilderImpl(String name, Supplier<C> previousBuilderSupplier, Neurons neurons) {
		super(name, previousBuilderSupplier, neurons);
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withGamma(WeightsMatrix gamma) {
		this.gamma = gamma;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withBeta(BiasMatrix beta) {
		this.beta = beta;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withMean(Matrix mean) {
		this.mean = mean;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withVariance(Matrix variance) {
		this.variance = variance;
		return this;
	}

	@Override
	public WeightsMatrix getGamma() {
		return gamma;
	}

	@Override
	public BiasMatrix getBeta() {
		return beta;
	}

	@Override
	public Matrix getMean() {
		return mean;
	}

	@Override
	public Matrix getVariance() {
		return variance;
	}

	@Override
	public C withConnectionToNeurons(Neurons neurons) {
		previousBuilderSupplier.get().getComponentsGraphNeurons().setRightNeurons(neurons);
		return previousBuilderSupplier.get();
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withConnectionWeights(WeightsMatrix connectionWeights) {
		previousBuilderSupplier.get().getBuilderState().setConnectionWeights(connectionWeights);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withBiasUnit() {
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withBiases(BiasMatrix biases) {
		previousBuilderSupplier.get().getBuilderState().setBiases(biases);
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withAxonsContextConfigurer(
			Consumer<AxonsContext> axonsContextConfigurer) {
		this.axonsContextConfigurer = axonsContextConfigurer;
		return this;
	}

	@Override
	public BatchNormDimension<Neurons> getBatchNormDimension() {
		return BatchNormDimension.INPUT_FEATURE;
	}
}
