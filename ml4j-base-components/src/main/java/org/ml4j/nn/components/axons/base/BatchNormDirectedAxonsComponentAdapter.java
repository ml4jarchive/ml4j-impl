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
package org.ml4j.nn.components.axons.base;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class BatchNormDirectedAxonsComponentAdapter<N extends Neurons> extends DefaultChainableDirectedComponentAdapter<DirectedAxonsComponentActivation, AxonsContext> 
	implements BatchNormDirectedAxonsComponent<N, Axons<N, N, ?>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public BatchNormDirectedAxonsComponentAdapter(
			BatchNormDirectedAxonsComponent<N, ?> delegated) {
		super(delegated, BatchNormDirectedAxonsComponent.class.getSimpleName() + ":" + ((Axons<N, N, ?>)((BatchNormDirectedAxonsComponent<N, ?>)delegated).getAxons()).getClass().getSimpleName());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Axons<N, N, ?> getAxons() {
		return (Axons<N, N, ?>)((BatchNormDirectedAxonsComponent<N, ?>)delegated).getAxons();
	}

	@SuppressWarnings("unchecked")
	@Override
	public BatchNormDirectedAxonsComponentAdapter<N> dup() {
		return new BatchNormDirectedAxonsComponentAdapter<N>( (BatchNormDirectedAxonsComponent<N, ?>) delegated.dup());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getExponentiallyWeightedAverageInputFeatureMeans();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getExponentiallyWeightedAverageInputFeatureVariances();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix meansColumnVector) {
		((BatchNormDirectedAxonsComponent<N, ?>)delegated).setExponentiallyWeightedAverageInputFeatureMeans(meansColumnVector);
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix variancesColumnVector) {
		((BatchNormDirectedAxonsComponent<N, ?>)delegated).setExponentiallyWeightedAverageInputFeatureVariances(variancesColumnVector);		
	}

	@SuppressWarnings("unchecked")
	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getBetaForExponentiallyWeightedAverages();
	}

}
