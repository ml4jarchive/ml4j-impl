package org.ml4j.nn.layers;

import java.util.Arrays;
import java.util.List;

import org.ml4j.EditableMatrix;

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

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.BiasMatrixImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.synapses.UndirectedSynapses;
import org.ml4j.nn.synapses.UndirectedSynapsesActivation;
import org.ml4j.nn.synapses.UndirectedSynapsesContext;
import org.ml4j.nn.synapses.UndirectedSynapsesImpl;
import org.ml4j.nn.synapses.UndirectedSynapsesInput;
import org.ml4j.nn.synapses.UndirectedSynapsesInputImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RestrictedBoltzmannLayerImpl implements RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(RestrictedBoltzmannLayerImpl.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private TrainableAxons<?, ?, ?> axons;

	private UndirectedSynapses<?, ?> synapses;

	/**
	 * @param axons                     The Axons.
	 * @param visibleActivationFunction The visible ActivationFunction
	 * @param hiddenActivationFunction  The hidden ActivationFunction
	 */
	public RestrictedBoltzmannLayerImpl(TrainableAxons<?, ?, ?> axons,
			ActivationFunction<?, ?> visibleActivationFunction, ActivationFunction<?, ?> hiddenActivationFunction) {
		this.axons = axons;
		this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
				hiddenActivationFunction);
	}

	/**
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param visibleNeurons            The visible Neurons.
	 * @param hiddenNeurons             The hidden Neurons.
	 * @param visibleActivationFunction The visible ActivationFunction.
	 * @param hiddenActivationFunction  The hidden ActivationFunction.
	 * @param matrixFactory             The MatrixFactory.
	 */
	public RestrictedBoltzmannLayerImpl(AxonsFactory axonsFactory, Neurons visibleNeurons, Neurons hiddenNeurons,
			ActivationFunction<?, ?> visibleActivationFunction, ActivationFunction<?, ?> hiddenActivationFunction,
			MatrixFactory matrixFactory) {
		this.axons = axonsFactory.createFullyConnectedAxons(new AxonsConfig<>(visibleNeurons, hiddenNeurons), 
				new WeightsMatrixImpl(null, new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
						Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), null);
		this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
				hiddenActivationFunction);
	}

	/**
	 * @param axonsFactory              A factory implementation to create axons.
	 * @param visibleNeurons            The visible Neurons.
	 * @param hiddenNeurons             The hidden Neurons.
	 * @param visibleActivationFunction The visible ActivationFunction.
	 * @param hiddenActivationFunction  The hidden ActivationFunction.
	 * @param matrixFactory             The MatrixFactory.
	 * @param initialWeights            The initial weights.
	 */
	public RestrictedBoltzmannLayerImpl(AxonsFactory axonsFactory, Neurons visibleNeurons, Neurons hiddenNeurons,
			ActivationFunction<?, ?> visibleActivationFunction, ActivationFunction<?, ?> hiddenActivationFunction,
			MatrixFactory matrixFactory, Matrix initialWeights, Matrix initialLeftToRightBiases,
			Matrix initialRightToLeftBiases) {
		this.axons = axonsFactory.createFullyConnectedAxons(new AxonsConfig<>(visibleNeurons, hiddenNeurons), 
				new WeightsMatrixImpl(initialWeights,
						new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE),
								WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),
				initialLeftToRightBiases == null ? null : new BiasMatrixImpl(initialLeftToRightBiases), 
						initialRightToLeftBiases == null? null : new BiasMatrixImpl(initialRightToLeftBiases));
		this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
				hiddenActivationFunction);
	}

	@Override
	public RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> dup() {
		return new RestrictedBoltzmannLayerImpl(axons.dup(), synapses.getLeftActivationFunction(),
				synapses.getRightActivationFunction());
	}

	@Override
	public TrainableAxons<?, ?, ?> getPrimaryAxons() {
		return axons;
	}

	/*
	 * @Override public List<UndirectedSynapses<?, ?>> getSynapses() { return
	 * Arrays.asList(synapses); }
	 */

	@Override
	public NeuronsActivation getOptimalVisibleActivationsForHiddenNeuron(int hiddenNeuronIndex,
			UndirectedLayerContext undirectedLayerContext, MatrixFactory matrixFactory) {
		LOGGER.debug("Obtaining optimal input for hidden neuron with index:" + hiddenNeuronIndex);
		Matrix weights = getPrimaryAxons().getDetachedAxonWeights().getConnectionWeights().getWeights();
		int countJ = weights.getColumns();
		float[] maximisingInputFeatures = new float[countJ];
		boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

		for (int j = 0; j < countJ; j++) {
			float wij = getWij(j, hiddenNeuronIndex, weights, hasBiasUnit);
			float sum = 0;

			if (wij != 0) {

				for (int j2 = 0; j2 < countJ; j2++) {
					float weight = getWij(j2, hiddenNeuronIndex, weights, hasBiasUnit);
					if (weight != 0) {
						sum = sum + (float) Math.pow(weight, 2);
					}
				}
				sum = (float) Math.sqrt(sum);
			}
			maximisingInputFeatures[j] = wij / sum;
		}
		return new NeuronsActivationImpl(getVisibleNeurons(),
				matrixFactory.createMatrixFromRows(new float[][] { maximisingInputFeatures }),
				NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	}

	private float getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
		int indICorrected = indI;
		return weights.get(indJ, indICorrected);
	}

	@Override
	public RestrictedBoltzmannLayerActivation activateHiddenNeuronsFromVisibleNeuronsData(
			NeuronsActivation visibleNeuronsActivation, UndirectedLayerContext layerContext) {
		UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(visibleNeuronsActivation);

		UndirectedSynapsesActivation hiddenNeuronsSynapseActivation = synapses.pushLeftToRight(synapsesInput, null,
				layerContext.createSynapsesContext(0));

		return new RestrictedBoltzmannLayerActivationImpl(hiddenNeuronsSynapseActivation, visibleNeuronsActivation,
				hiddenNeuronsSynapseActivation.getOutput());
	}

	@Override
	public RestrictedBoltzmannLayerActivation activateHiddenNeuronsFromVisibleNeuronsReconstruction(
			RestrictedBoltzmannLayerActivation visibleNeuronsReconstruction, UndirectedLayerContext layerContext) {
		UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(
				new NeuronsActivationImpl(getHiddenNeurons(),
						visibleNeuronsReconstruction.getSynapsesActivation().getOutput()
								.getActivations(layerContext.getMatrixFactory()),
								NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET));

		UndirectedSynapsesActivation hiddenNeuronsSynapseActivation = synapses.pushLeftToRight(synapsesInput,
				visibleNeuronsReconstruction.getSynapsesActivation(), layerContext.createSynapsesContext(0));

		return new RestrictedBoltzmannLayerActivationImpl(hiddenNeuronsSynapseActivation,
				visibleNeuronsReconstruction.getVisibleActivationProbablities(),
				hiddenNeuronsSynapseActivation.getOutput());
	}

	@Override
	public RestrictedBoltzmannLayerActivation activateVisibleNeuronsFromHiddenNeurons(
			NeuronsActivation hiddenNeuronsDataActivation, UndirectedLayerContext layerContext) {
		UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(hiddenNeuronsDataActivation);
		UndirectedSynapsesContext context = layerContext.createSynapsesContext(0);

		UndirectedSynapsesActivation visibleNeuronsSynapseActivation = synapses.pushRightToLeft(synapsesInput, null,
				context);

		return new RestrictedBoltzmannLayerActivationImpl(visibleNeuronsSynapseActivation,
				visibleNeuronsSynapseActivation.getOutput(), hiddenNeuronsDataActivation);

	}

	@Override
	public RestrictedBoltzmannLayerActivation activateVisibleNeuronsFromHiddenNeuronsProbabilities(
			RestrictedBoltzmannLayerActivation previousVisibleToHiddenNeuronsActivation,
			UndirectedLayerContext layerContext) {
		UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(
				previousVisibleToHiddenNeuronsActivation.getHiddenActivationProbabilities());
		UndirectedSynapsesContext context = layerContext.createSynapsesContext(0);

		UndirectedSynapsesActivation visibleNeuronsSynapseActivation = synapses.pushRightToLeft(synapsesInput,
				previousVisibleToHiddenNeuronsActivation.getSynapsesActivation(), context);

		return new RestrictedBoltzmannLayerActivationImpl(visibleNeuronsSynapseActivation,
				visibleNeuronsSynapseActivation.getOutput(),
				previousVisibleToHiddenNeuronsActivation.getHiddenActivationProbabilities());
	}

	@Override
	public RestrictedBoltzmannLayerActivation activateVisibleNeuronsFromHiddenNeuronsSample(
			RestrictedBoltzmannLayerActivation previousVisibleToHiddenNeuronsActivation,
			UndirectedLayerContext layerContext) {

		NeuronsActivation sample = previousVisibleToHiddenNeuronsActivation
				.getHiddenActivationBinarySample(layerContext.getMatrixFactory());

		UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(
				new NeuronsActivationImpl(getHiddenNeurons(), sample.getActivations(layerContext.getMatrixFactory()),
						NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET));
		UndirectedSynapsesContext context = layerContext.createSynapsesContext(0);
		UndirectedSynapsesActivation visibleNeuronsSynapseActivation = synapses.pushRightToLeft(synapsesInput,
				previousVisibleToHiddenNeuronsActivation.getSynapsesActivation(), context);

		return new RestrictedBoltzmannLayerActivationImpl(visibleNeuronsSynapseActivation,
				new NeuronsActivationImpl(getVisibleNeurons(),
						visibleNeuronsSynapseActivation.getOutput().getActivations(layerContext.getMatrixFactory()),
						NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET),
				previousVisibleToHiddenNeuronsActivation.getHiddenActivationProbabilities());
	}

	@Override
	public Neurons getHiddenNeurons() {
		return axons.getRightNeurons();
	}

	@Override
	public Neurons getVisibleNeurons() {
		return axons.getLeftNeurons();
	}

	/**
	 * @param data           The training data.
	 * @param visibleNeurons The visible neurons.
	 * @param hiddenNeurons  The hidden neurons.
	 * @param learningRate   The learning rate.
	 * @param matrixFactory  The matrix factory.
	 * @return The initial connection weights.
	 */
	public static Matrix generateInitialConnectionWeights(NeuronsActivation data, Neurons visibleNeurons,
			Neurons hiddenNeurons, float learningRate, MatrixFactory matrixFactory) {

		int initialHiddenUnitBiasWeight = -4;
		EditableMatrix thetas = matrixFactory
				.createRandn(visibleNeurons.getNeuronCountIncludingBias(), hiddenNeurons.getNeuronCountIncludingBias())
				.mul(learningRate).asEditableMatrix();
		for (int i = 1; i < thetas.getColumns(); i++) {
			thetas.put(0, i, initialHiddenUnitBiasWeight);
		}
		for (int i = 1; i < thetas.getRows(); i++) {
			double[] proportionsOfOnUnits = getProportionsOfOnUnits(data.getActivations(matrixFactory));
			double proportionOfTimeUnitActivated = proportionsOfOnUnits[i - 1];
			// Needed to add the following to limit p here, otherwise the log blows up
			proportionOfTimeUnitActivated = Math.max(proportionOfTimeUnitActivated, 0.001);
			float initialVisibleUnitBiasWeight = (float) Math
					.log(proportionOfTimeUnitActivated / (1 - proportionOfTimeUnitActivated));
			thetas.put(i, 0, initialVisibleUnitBiasWeight);
		}
		thetas.put(0, 0, 0);
		return thetas;
	}

	private static double[] getProportionsOfOnUnits(Matrix data) {
		int[] counts = new int[data.getColumns()];
		for (int row = 0; row < data.getRows(); row++) {
			float[] dat = data.getRow(row).getRowByRowArray();
			for (int i = 0; i < counts.length; i++) {
				if (dat[i] == 1) {
					counts[i]++;
				}
			}
		}
		double[] props = new double[counts.length];
		for (int i = 0; i < props.length; i++) {
			props[i] = counts[i] / data.getColumns();
		}
		return props;
	}

	@Override
	public List<UndirectedSynapses<?, ?>> getComponents() {
		return Arrays.asList(synapses);
	}

}
