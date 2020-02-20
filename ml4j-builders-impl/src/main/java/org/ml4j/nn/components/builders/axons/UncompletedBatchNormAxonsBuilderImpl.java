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

import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsContextConfigurer;
import org.ml4j.nn.axons.BatchNormConfig.BatchNormDimension;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsVector;
import org.ml4j.nn.neurons.Neurons;

public class UncompletedBatchNormAxonsBuilderImpl<C extends AxonsBuilder<?>>
		extends UncompletedAxonsBuilderImpl<Neurons, C> implements UncompletedBatchNormAxonsBuilder<Neurons, C> {

	private WeightsVector gamma;
	private BiasVector beta;
	private FeaturesVector mean;
	private FeaturesVector variance;

	public UncompletedBatchNormAxonsBuilderImpl(String name, Supplier<C> previousBuilderSupplier, Neurons neurons) {
		super(name, previousBuilderSupplier, neurons);
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withGamma(WeightsVector gamma) {
		this.gamma = gamma;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withBeta(BiasVector beta) {
		this.beta = beta;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withMean(FeaturesVector mean) {
		this.mean = mean;
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withVariance(FeaturesVector variance) {
		this.variance = variance;
		return this;
	}

	@Override
	public WeightsVector getGamma() {
		return gamma;
	}

	@Override
	public BiasVector getBeta() {
		return beta;
	}

	@Override
	public FeaturesVector getMean() {
		return mean;
	}

	@Override
	public FeaturesVector getVariance() {
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
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withBiases(BiasVector biases) {
		previousBuilderSupplier.get().getBuilderState().setBiases(biases);
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<Neurons, C> withAxonsContextConfigurer(
			AxonsContextConfigurer axonsContextConfigurer) {
		this.axonsContextConfigurer = axonsContextConfigurer;
		return this;
	}

	@Override
	public BatchNormDimension<Neurons> getBatchNormDimension() {
		return BatchNormDimension.INPUT_FEATURE;
	}
}
