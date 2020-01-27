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

package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.synapses.DirectedSynapsesImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A default base implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of primary Axons in this FeedForwardLayer.
 */
public abstract class FeedForwardLayerBase<A extends Axons<?, ?, ?>, L extends FeedForwardLayer<A, L>>
		extends AbstractFeedForwardLayer<A, L> implements FeedForwardLayer<A, L> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(FeedForwardLayerBase.class);

	protected A primaryAxons;

	protected DifferentiableActivationFunction primaryActivationFunction;

	protected boolean withBatchNorm;

	protected DirectedComponentFactory directedComponentFactory;
	
	/**
	 * @param primaryAxons       The primary Axons
	 * @param activationFunction The primary activation function
	 * @param matrixFactory      The matrix factory
	 * @param withBatchNorm      Whether to enable batch norm.
	 */
	protected FeedForwardLayerBase(String name, DirectedComponentFactory directedComponentFactory, A primaryAxons,
			DifferentiableActivationFunction activationFunction, MatrixFactory matrixFactory, boolean withBatchNorm) {
		super(name, directedComponentFactory, directedComponentFactory.createDirectedComponentChain(
				getSynapses(name, directedComponentFactory, matrixFactory, primaryAxons, activationFunction, withBatchNorm)),
				matrixFactory);
		this.primaryAxons = primaryAxons;
		this.primaryActivationFunction = activationFunction;
		this.matrixFactory = matrixFactory;
		this.withBatchNorm = withBatchNorm;
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public int getInputNeuronCount() {
		return primaryAxons.getLeftNeurons().getNeuronCountIncludingBias();
	}

	@Override
	public int getOutputNeuronCount() {
		return primaryAxons.getRightNeurons().getNeuronCountIncludingBias();
	}

	@Override
	public A getPrimaryAxons() {
		return primaryAxons;
	}

	@Override
	public DifferentiableActivationFunction getPrimaryActivationFunction() {
		return primaryActivationFunction;
	}

	protected static List<DefaultChainableDirectedComponent<?, ?>> getSynapses(String name, 
			DirectedComponentFactory directedComponentFactory, MatrixFactory matrixFactory, Axons<?, ?, ?> primaryAxons,
			DifferentiableActivationFunction primaryActivationFunction, boolean withBatchNorm) {
		Objects.requireNonNull(primaryAxons, "primaryAxons");
		List<DefaultChainableDirectedComponent<?, ?>> synapses = new ArrayList<>();
		if (withBatchNorm) {
			throw new UnsupportedOperationException("Not yet implemented");
		} else {
			synapses.add(new DirectedSynapsesImpl<>(name + ":DirectedSynapses", directedComponentFactory, primaryAxons, primaryActivationFunction));
		}
		return synapses;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		List<DefaultChainableDirectedComponent<?, ?>> components = new ArrayList<>();
		Objects.requireNonNull(primaryAxons, "primaryAxons");
		components.addAll(getSynapses(name, directedComponentFactory, matrixFactory, primaryAxons, primaryActivationFunction,
				withBatchNorm));
		return components;
	}

	protected TrailingActivationFunctionDirectedComponentChain createChain() {

		DefaultDirectedComponentChain synapseChain = directedComponentFactory.createDirectedComponentChain(getSynapses(name, 
				directedComponentFactory, matrixFactory, primaryAxons, primaryActivationFunction, withBatchNorm));

		List<DefaultChainableDirectedComponent<? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> chainableComponents = new ArrayList<>();
		chainableComponents.addAll(synapseChain.decompose());
		return new TrailingActivationFunctionDirectedComponentChainImpl(directedComponentFactory, chainableComponents);
	}

	@Override
	public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
			DirectedLayerContext directedLayerContext) {
		LOGGER.debug(directedLayerContext.toString() + ":Forward propagating through layer");

		DirectedComponentsContext componentsContext = new DirectedComponentsContextImpl(
				directedLayerContext.getMatrixFactory(), directedLayerContext.isTrainingContext());

		TrailingActivationFunctionDirectedComponentChainActivation activation = createChain()
				.forwardPropagate(inputNeuronsActivation, componentsContext);

		return new DirectedLayerActivationImpl(this, activation, directedLayerContext);
	}

	@Override
	public DirectedLayerContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext.getContext(this, () -> new DirectedLayerContextImpl(
				matrixFactory, directedComponentsContext.isTrainingContext()));
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return getComponents().stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronIndex,
			DirectedLayerContext directedLayerContext) {
		LOGGER.debug("Obtaining optimal input for output neuron with index:" + outputNeuronIndex);
		if (!(getPrimaryAxons() instanceof TrainableAxons)) {
			throw new UnsupportedOperationException("Axons do not have connection weights");
		}
		Matrix weightsOnly = ((TrainableAxons<?, ?, ?>) getPrimaryAxons()).getDetachedAxonWeights()
				.getConnectionWeights();

		int countJ = weightsOnly.getColumns(); // - (getPrimaryAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0);
		float[] maximisingInputFeatures = new float[countJ];
		boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

		for (int j = 0; j < countJ; j++) {
			float wij = getWij(j, outputNeuronIndex, weightsOnly, hasBiasUnit);
			float sum = 0;

			if (wij != 0) {

				for (int j2 = 0; j2 < countJ; j2++) {
					double weight = getWij(j2, outputNeuronIndex, weightsOnly, hasBiasUnit);
					if (weight != 0) {
						sum = sum + (float) Math.pow(weight, 2);
					}
				}
				sum = (float) Math.sqrt(sum);
			}
			maximisingInputFeatures[j] = wij / sum;
		}
		return new NeuronsActivationImpl(getInputNeurons(),
				directedLayerContext.getMatrixFactory().createMatrixFromRows(new float[][] { maximisingInputFeatures }),
				NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET);
	}

	private float getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
		int indICorrected = indI; // + (hasBiasUnit ? 1 : 0);
		return weights.get(indJ, indICorrected);
	}

	@Override
	public Neurons getInputNeurons() {
		return trailingActivationFunctionDirectedComponentChain.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return trailingActivationFunctionDirectedComponentChain.getOutputNeurons();
	}

}
