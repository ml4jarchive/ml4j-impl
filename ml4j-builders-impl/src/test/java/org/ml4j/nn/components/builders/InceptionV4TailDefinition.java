/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.components.builders;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * @author Michael Lavelle
 */
public class InceptionV4TailDefinition implements Component3DtoNon3DGraphDefinition {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private InceptionV4WeightsLoader weightsLoader;
	private float regularisationLambda;
	private float dropoutKeepProbability;

	public InceptionV4TailDefinition(InceptionV4WeightsLoader weightsLoader) {
		this.weightsLoader = weightsLoader;
		this.dropoutKeepProbability = 1f;
	}

	@Override
	public Neurons3D getInputNeurons() {
		return new Neurons3D(8, 8, 1536, false);
	}
	
	@Override
	public Neurons getOutputNeurons() {
		return new Neurons(1001, false);
	}

	public <T extends NeuralComponent> InitialComponentsGraphBuilder<T> createComponentGraph(
			InitialComponents3DGraphBuilder<T> start, NeuralComponentFactory<T> neuralComponentFactory) {
		return start
					.withAveragePoolingAxons("average_pooling_5")
						.withStride(1, 1).withFilterSize(8, 8).withValidPadding()
						.withConnectionToNeurons(new Neurons3D(1, 1, 1536, false))
					.withFullyConnectedAxons("dense_1")
						.withConnectionWeights(weightsLoader.getDenseLayerWeights("dense_1_kernel0", 1001, 1536))
						.withBiasUnit()
						.withBiases(weightsLoader.getDenseLayerBiases("dense_1_bias0", 1001, 1))
						.withAxonsContextConfigurer(c -> c.withRegularisationLambda(regularisationLambda).withLeftHandInputDropoutKeepProbability(dropoutKeepProbability))
					.withConnectionToNeurons(new Neurons(1001, false))
					.withActivationFunction("softmax_1", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SOFTMAX), new ActivationFunctionProperties());
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.getBaseType(NeuralComponentBaseType.DEFINITION);
	}

	@Override
	public String getName() {
		return "inceptionv4_tail";
	}

	public void setRegularisationLambda(float regularisationLambda) {
		this.regularisationLambda = regularisationLambda;
	}

	public void setDropoutKeepProbability(float dropoutKeepProbability) {
		this.dropoutKeepProbability = dropoutKeepProbability;
	}
}
