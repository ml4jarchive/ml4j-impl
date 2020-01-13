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
package org.ml4j.nn.components.factories;

import java.util.List;
import java.util.function.IntSupplier;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.base.BatchNormDirectedAxonsComponentAdapter;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentAdapter;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentAdapter;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentAdapter;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class DirectedComponentFactoryAdapter implements DirectedComponentFactory {

	protected DirectedComponentFactory delegated;
	
	
	public DirectedComponentFactoryAdapter(DirectedComponentFactory delegated) {
		this.delegated = delegated;
	}
	
	@Override
	public DirectedAxonsComponent<Neurons, Neurons, ?> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return new DirectedAxonsComponentAdapter<>(delegated.createFullyConnectedAxonsComponent(leftNeurons, 
				rightNeurons, connectionWeights, biases));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config,
			Matrix connectionWeights, Matrix biases) {
		return new DirectedAxonsComponentAdapter<>(delegated.createConvolutionalAxonsComponent(leftNeurons, rightNeurons, config,
				connectionWeights, biases));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config,
			boolean scaleOutputs) {
		return new DirectedAxonsComponentAdapter<>(delegated.createMaxPoolingAxonsComponent(leftNeurons, rightNeurons, 
				config, scaleOutputs));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config) {
		return new DirectedAxonsComponentAdapter<>(delegated.createAveragePoolingAxonsComponent(leftNeurons, rightNeurons, 
				config));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return new BatchNormDirectedAxonsComponentAdapter<>(delegated.createBatchNormAxonsComponent(leftNeurons, rightNeurons));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		return new BatchNormDirectedAxonsComponentAdapter<>(delegated.createBatchNormAxonsComponent(leftNeurons, rightNeurons, gamma, beta, mean, stddev));
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		return new BatchNormDirectedAxonsComponentAdapter<>(delegated.createConvolutionalBatchNormAxonsComponent(leftNeurons, rightNeurons));
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		return new BatchNormDirectedAxonsComponentAdapter<>(delegated.createConvolutionalBatchNormAxonsComponent(leftNeurons, rightNeurons, gamma, beta, mean, stddev));
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(
			Axons<L, R, ?> axons) {
		return new DirectedAxonsComponentAdapter<>(delegated.createDirectedAxonsComponent(axons));
	}

	@Override
	public <N extends Neurons> DirectedAxonsComponent<N, N, ?> createPassThroughAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return delegated.createPassThroughAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(IntSupplier targetComponentsCount) {
		return new OneToManyDirectedComponentAdapter<>(delegated.createOneToManyDirectedComponent(targetComponentsCount));
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			PathCombinationStrategy pathCombinationStrategy) {
		return new ManyToOneDirectedComponentAdapter<>(delegated.createManyToOneDirectedComponent(pathCombinationStrategy));
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons,
			DifferentiableActivationFunction differentiableActivationFunction) {
		return delegated.createDifferentiableActivationFunctionComponent(neurons, differentiableActivationFunction);
	}
	
	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons,
			ActivationFunctionType activationFunctionType) {
		return delegated.createDifferentiableActivationFunctionComponent(neurons, activationFunctionType);
	}

	@Override
	public DefaultDirectedComponentChain createDirectedComponentChain(
			List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		return delegated.createDirectedComponentChain(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(Neurons arg0, Neurons arg1,
			List<DefaultChainableDirectedComponent<?, ?>> parallelComponents, PathCombinationStrategy arg3) {
		return delegated.createDirectedComponentBipoleGraph(arg0, arg1, parallelComponents, arg3);
	}

	@Override
	public <S extends DefaultChainableDirectedComponent<?, ?>> DefaultChainableDirectedComponent<?, ?> createComponent(
			Neurons leftNeurons, Neurons rightNeurons, NeuralComponentType<S> neuralComponentType) {
		return delegated.createComponent(leftNeurons, rightNeurons, neuralComponentType);
	}
}
