/*
 * Copyright 2017 the original author or authors.
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

package org.ml4j.nn.supervised;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.CostAndGradientsImpl;
import org.ml4j.nn.FeedForwardNeuralNetworkBase;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * Default implementation of SupervisedFeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public class SupervisedFeedForwardNeuralNetworkImpl extends
		FeedForwardNeuralNetworkBase<FeedForwardNeuralNetworkContext, DefaultDirectedComponentChain, SupervisedFeedForwardNeuralNetwork>
		implements SupervisedFeedForwardNeuralNetwork {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
	 * 
	 * @param layers The layers
	 */
	public SupervisedFeedForwardNeuralNetworkImpl(String name, DirectedComponentFactory directedComponentFactory,
			DefaultDirectedComponentChain initialisingComponentChain) {
		super(name, directedComponentFactory, initialisingComponentChain);
	}

	protected SupervisedFeedForwardNeuralNetworkImpl(String name, DefaultDirectedComponentChain initialisingComponentChain,
			TrailingActivationFunctionDirectedComponentChain trailingActivationFunctionComponentChain) {
		super(name, initialisingComponentChain, trailingActivationFunctionComponentChain);
	}

	/**
	 * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
	 * 
	 * @param layers The layers
	 */
	public SupervisedFeedForwardNeuralNetworkImpl(String name, DirectedComponentFactory directedComponentFactory,
			List<DefaultChainableDirectedComponent<?, ?>> componentList) {
		super(name, directedComponentFactory, directedComponentFactory.createDirectedComponentChain(componentList));
	}

	/**
	 * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
	 * 
	 * @param layers The layers
	 */
	@SafeVarargs
	public SupervisedFeedForwardNeuralNetworkImpl(String name, DirectedComponentFactory directedComponentFactory,
			DefaultChainableDirectedComponent<?, ?>... componentList) {
		super(name, directedComponentFactory,
				directedComponentFactory.createDirectedComponentChain(Arrays.asList(componentList)));
	}

	@Override
	public void train(NeuronsActivation trainingDataActivations, NeuronsActivation trainingLabelActivations,
			FeedForwardNeuralNetworkContext trainingContext) {
		if (trainingDataActivations
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		if (trainingLabelActivations
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}
		super.train(trainingDataActivations, trainingLabelActivations, trainingContext);
	}

	@Override
	public void train(Stream<LabeledData<NeuronsActivation, NeuronsActivation>> trainingDataActivations,
			FeedForwardNeuralNetworkContext trainingContext) {
		super.train(trainingDataActivations, trainingContext);
	}

	@Override
	public void train(Supplier<Stream<LabeledData<NeuronsActivation, NeuronsActivation>>> trainingDataActivations,
			FeedForwardNeuralNetworkContext trainingContext, Consumer<Float> onEpochAverageCostHandler) {
		super.train(trainingDataActivations, trainingContext, onEpochAverageCostHandler);
	}

	/**
	 * Return the prediction accuracy.
	 * 
	 * @param inputActivations                 The input activations.
	 * @param desiredClassificationActivations The desired prediction activations.
	 * @param context                          The context.
	 * @return The accuracy
	 */
	@Override
	public float getClassificationAccuracy(NeuronsActivation inputActivations,
			NeuronsActivation desiredClassificationActivations, FeedForwardNeuralNetworkContext context) {

		// Forward propagate the trainingDataActivations
		ForwardPropagation forwardPropagation = forwardPropagate(inputActivations, context);

		Matrix predictions = getClassifications(
				forwardPropagation.getOutput().getActivations(context.getMatrixFactory()).transpose(),
				context.getMatrixFactory());

		return computeAccuracy(predictions,
				desiredClassificationActivations.getActivations(context.getMatrixFactory()).transpose());
	}

	private Matrix getClassifications(Matrix outputActivations, MatrixFactory matrixFactory) {

		EditableMatrix predictions = matrixFactory
				.createZeros(outputActivations.getRows(), outputActivations.getColumns()).asEditableMatrix();
		for (int row = 0; row < outputActivations.getRows(); row++) {

			int index = outputActivations.getRow(row).argmax();
			predictions.put(row, index, 1);
		}
		return predictions;
	}

	/**
	 * Helper function to compute the accuracy of predictions using calculated
	 * predictions predictions and correct output matrix.
	 *
	 * @param predictions The predictions
	 * @param Y           The desired output labels
	 * @return The accuracy of the network
	 */
	protected float computeAccuracy(Matrix predictions, Matrix outputs) {
		return ((predictions.mul(outputs)).sum()) * 100 / outputs.getRows();
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork dup(DirectedComponentFactory directedComponentFactory) {
		return new SupervisedFeedForwardNeuralNetworkImpl(name, initialisingComponentChain.dup(directedComponentFactory),
				trailingActivationFunctionComponentChain.dup(directedComponentFactory));
	}

	@Override
	public CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
			NeuronsActivation desiredOutpuActivations, FeedForwardNeuralNetworkContext trainingContext) {
		return super.getCostAndGradients(inputActivations, desiredOutpuActivations, trainingContext);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return trailingActivationFunctionComponentChain.decompose();
	}

	@Override
	public FeedForwardNeuralNetworkContext getContext(DirectedComponentsContext directedComponentsContext) {
		return new FeedForwardNeuralNetworkContextImpl(directedComponentsContext, directedComponentsContext.isTrainingContext());
	}

	@Override
	public Neurons getInputNeurons() {
		return trailingActivationFunctionComponentChain.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return trailingActivationFunctionComponentChain.getOutputNeurons();
	}

	@Override
	public String toString() {
		return "SupervisedFeedForwardNeuralNetworkImpl [name='" + name + "', inputNeurons=" + getInputNeurons()
				+ ", outputNeurons=" + getOutputNeurons() + ", costFunction=" + getCostFunction()
				+ ", lastEpochTrainingContext=" + getLastEpochTrainingContext() + ", componentType="
				+ getComponentType() + ", optimisedFor=" + optimisedFor() + "]";
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		return initialisingComponentChain.getComponents();
	}

	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		allComponentsIncludingThis.addAll(trailingActivationFunctionComponentChain.flatten());
		return allComponentsIncludingThis;
	}


}
