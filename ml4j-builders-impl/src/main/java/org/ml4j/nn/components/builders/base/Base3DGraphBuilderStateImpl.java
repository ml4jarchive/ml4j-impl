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
package org.ml4j.nn.components.builders.base;

import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedPoolingAxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class Base3DGraphBuilderStateImpl implements Base3DGraphBuilderState {

	protected ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons;
	protected UncompletedConvolutionalAxonsBuilder<?> convolutionalAxonsBuilder;
	protected UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder;

	protected UncompletedPoolingAxonsBuilder<?> maxPoolingAxonsBuilder;
	protected UncompletedPoolingAxonsBuilder<?> averagePoolingAxonsBuilder;
	protected UncompletedBatchNormAxonsBuilder<?> batchNormAxonsBuilder;

	protected SynapsesAxons3DGraphBuilder<?, ?, ?> synapsesBuilder;
	private WeightsMatrix connectionWeights;
	private BiasMatrix biases;

	public Base3DGraphBuilderStateImpl() {
	}

	public Base3DGraphBuilderStateImpl(Neurons3D currentNeurons) {
		this.componentsGraphNeurons = new ComponentsGraphNeuronsImpl<>(currentNeurons);
	}

	@Override
	public BaseGraphBuilderStateImpl getNon3DBuilderState() {
		BaseGraphBuilderStateImpl builderState = new BaseGraphBuilderStateImpl();
		builderState.setConnectionWeights(connectionWeights);
		ComponentsGraphNeurons<Neurons> non3DcomponentsGraphNeurons = new ComponentsGraphNeuronsImpl<>(
				componentsGraphNeurons.getCurrentNeurons(), componentsGraphNeurons.getRightNeurons());
		builderState.setComponentsGraphNeurons(non3DcomponentsGraphNeurons);
		builderState.setFullyConnectedAxonsBuilder(fullyConnectedAxonsBuilder);
		return builderState;
	}

	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}

	public void setComponentsGraphNeurons(ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons) {
		this.componentsGraphNeurons = componentsGraphNeurons;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<?> getConvolutionalAxonsBuilder() {
		return convolutionalAxonsBuilder;
	}

	@Override
	public void setConvolutionalAxonsBuilder(UncompletedConvolutionalAxonsBuilder<?> convolutionalAxonsBuilder) {
		this.convolutionalAxonsBuilder = convolutionalAxonsBuilder;
	}

	@Override
	public UncompletedPoolingAxonsBuilder<?> getMaxPoolingAxonsBuilder() {
		return maxPoolingAxonsBuilder;
	}

	@Override
	public void setMaxPoolingAxonsBuilder(UncompletedPoolingAxonsBuilder<?> maxPoolingAxonsBuilder) {
		this.maxPoolingAxonsBuilder = maxPoolingAxonsBuilder;
	}

	@Override
	public SynapsesAxons3DGraphBuilder<?, ?, ?> getSynapsesBuilder() {
		return synapsesBuilder;
	}

	@Override
	public void setSynapsesBuilder(SynapsesAxons3DGraphBuilder<?, ?, ?> synapsesBuilder) {
		this.synapsesBuilder = synapsesBuilder;
	}

	@Override
	public WeightsMatrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public void setConnectionWeights(WeightsMatrix connectionWeights) {
		this.connectionWeights = connectionWeights;
	}

	@Override
	public BiasMatrix getBiases() {
		return biases;
	}

	@Override
	public void setBiases(BiasMatrix biases) {
		this.biases = biases;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<?> getFullyConnectedAxonsBuilder() {
		return fullyConnectedAxonsBuilder;
	}

	@Override
	public void setFullyConnectedAxonsBuilder(UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder) {
		this.fullyConnectedAxonsBuilder = fullyConnectedAxonsBuilder;
	}

	@Override
	public void setAveragePoolingAxonsBuilder(UncompletedPoolingAxonsBuilder<?> axonsBuilder) {
		this.averagePoolingAxonsBuilder = axonsBuilder;
	}

	@Override
	public UncompletedPoolingAxonsBuilder<?> getAveragePoolingAxonsBuilder() {
		return averagePoolingAxonsBuilder;
	}

	@Override
	public void setBatchNormAxonsBuilder(UncompletedBatchNormAxonsBuilder<?> axonsBuilder) {
		this.batchNormAxonsBuilder = axonsBuilder;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<?> getBatchNormAxonsBuilder() {
		return batchNormAxonsBuilder;
	}

}
