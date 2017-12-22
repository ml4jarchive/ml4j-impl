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

package org.ml4j.nn.layers;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.UndirectedSynapses;
import org.ml4j.nn.synapses.UndirectedSynapsesActivation;
import org.ml4j.nn.synapses.UndirectedSynapsesContext;
import org.ml4j.nn.synapses.UndirectedSynapsesImpl;
import org.ml4j.nn.synapses.UndirectedSynapsesInput;
import org.ml4j.nn.synapses.UndirectedSynapsesInputImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class RestrictedBoltzmannLayerImpl implements RestrictedBoltzmannLayer<FullyConnectedAxons> {

  private static final Logger LOGGER = LoggerFactory.getLogger(RestrictedBoltzmannLayerImpl.class);

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private FullyConnectedAxons axons;

  private UndirectedSynapses<?, ?> synapses;

  /**
   * @param axons The Axons.
   * @param visibleActivationFunction The visible ActivationFunction
   * @param hiddenActivationFunction The hidden ActivationFunction
   */
  public RestrictedBoltzmannLayerImpl(FullyConnectedAxons axons,
      ActivationFunction<?, ?> visibleActivationFunction, 
      ActivationFunction<?, ?> hiddenActivationFunction) {
    this.axons = axons;
    this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
        hiddenActivationFunction);
  }

  /**
   * @param visibleNeurons The visible Neurons.
   * @param hiddenNeurons The hidden Neurons.
   * @param visibleActivationFunction The visible ActivationFunction.
   * @param hiddenActivationFunction The hidden ActivationFunction.
   * @param matrixFactory The MatrixFactory.
   */
  public RestrictedBoltzmannLayerImpl(Neurons visibleNeurons, Neurons hiddenNeurons,
      ActivationFunction<? ,?> visibleActivationFunction, 
      ActivationFunction<?, ?> hiddenActivationFunction,
      MatrixFactory matrixFactory) {
    this.axons = new FullyConnectedAxonsImpl(visibleNeurons, hiddenNeurons, matrixFactory);
    this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
        hiddenActivationFunction);
  }

  /**
   * @param visibleNeurons The visible Neurons.
   * @param hiddenNeurons The hidden Neurons.
   * @param visibleActivationFunction The visible ActivationFunction.
   * @param hiddenActivationFunction The hidden ActivationFunction.
   * @param matrixFactory The MatrixFactory.
   * @param initialWeights The initial weights.
   */
  public RestrictedBoltzmannLayerImpl(Neurons visibleNeurons, Neurons hiddenNeurons,
      ActivationFunction<?, ?> visibleActivationFunction, 
      ActivationFunction<?, ?> hiddenActivationFunction,
      MatrixFactory matrixFactory, Matrix initialWeights) {
    this.axons =
        new FullyConnectedAxonsImpl(visibleNeurons, hiddenNeurons, matrixFactory, initialWeights);
    this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
        hiddenActivationFunction);
  }

  @Override
  public RestrictedBoltzmannLayer<FullyConnectedAxons> dup() {
    return new RestrictedBoltzmannLayerImpl(axons.dup(), synapses.getLeftActivationFunction(),
        synapses.getRightActivationFunction());
  }

  @Override
  public FullyConnectedAxons getPrimaryAxons() {
    return axons;
  }

  @Override
  public List<UndirectedSynapses<?, ?>> getSynapses() {
    return Arrays.asList(synapses);
  }

  @Override
  public NeuronsActivation getOptimalVisibleActivationsForHiddenNeuron(int hiddenNeuronIndex,
      UndirectedLayerContext undirectedLayerContext) {
    LOGGER.debug("Obtaining optimal input for hidden neuron with index:" + hiddenNeuronIndex);
    Matrix weights = getPrimaryAxons().getDetachedConnectionWeights();
    int countJ = weights.getRows() - (getPrimaryAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0);
    double[] maximisingInputFeatures = new double[countJ];
    boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

    for (int j = 0; j < countJ; j++) {
      double wij = getWij(j, hiddenNeuronIndex, weights, hasBiasUnit);
      double sum = 0;

      if (wij != 0) {

        for (int j2 = 0; j2 < countJ; j2++) {
          double weight = getWij(j2, hiddenNeuronIndex, weights, hasBiasUnit);
          if (weight != 0) {
            sum = sum + Math.pow(weight, 2);
          }
        }
        sum = Math.sqrt(sum);
      }
      maximisingInputFeatures[j] = wij / sum;
    }
    return new NeuronsActivation(
        undirectedLayerContext.getMatrixFactory()
            .createMatrix(new double[][] {maximisingInputFeatures}),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  private double getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
    int indICorrected = indI + (hasBiasUnit ? 1 : 0);
    return weights.get(indICorrected, indJ);
  }

  @Override
  public RestrictedBoltzmannLayerActivation activateHiddenNeuronsFromVisibleNeuronsData(
      NeuronsActivation visibleNeuronsActivation, UndirectedLayerContext layerContext) {
    UndirectedSynapsesInput synapsesInput =
        new UndirectedSynapsesInputImpl(visibleNeuronsActivation);

    UndirectedSynapsesActivation hiddenNeuronsSynapseActivation =
        synapses.pushLeftToRight(synapsesInput, null, layerContext.createSynapsesContext(0));

    return new RestrictedBoltzmannLayerActivationImpl(hiddenNeuronsSynapseActivation,
        visibleNeuronsActivation, hiddenNeuronsSynapseActivation.getOutput());
  }

  @Override
  public RestrictedBoltzmannLayerActivation activateHiddenNeuronsFromVisibleNeuronsReconstruction(
      RestrictedBoltzmannLayerActivation visibleNeuronsReconstruction,
      UndirectedLayerContext layerContext) {
    UndirectedSynapsesInput synapsesInput = new UndirectedSynapsesInputImpl(
        new NeuronsActivation(
            visibleNeuronsReconstruction.getSynapsesActivation().getOutput()
            .getActivations().transpose(), 
            NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET));

    UndirectedSynapsesActivation hiddenNeuronsSynapseActivation = synapses.pushLeftToRight(
        synapsesInput, visibleNeuronsReconstruction.getSynapsesActivation(),
        layerContext.createSynapsesContext(0));

    return new RestrictedBoltzmannLayerActivationImpl(hiddenNeuronsSynapseActivation,
        visibleNeuronsReconstruction.getVisibleActivationProbablities(),
        hiddenNeuronsSynapseActivation.getOutput());
  }

  @Override
  public RestrictedBoltzmannLayerActivation activateVisibleNeuronsFromHiddenNeurons(
      NeuronsActivation hiddenNeuronsDataActivation, UndirectedLayerContext layerContext) {
    UndirectedSynapsesInput synapsesInput =
        new UndirectedSynapsesInputImpl(hiddenNeuronsDataActivation);
    UndirectedSynapsesContext context = layerContext.createSynapsesContext(0);

    UndirectedSynapsesActivation visibleNeuronsSynapseActivation =
        synapses.pushRightToLeft(synapsesInput, null, context);

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

    UndirectedSynapsesActivation visibleNeuronsSynapseActivation = synapses.pushRightToLeft(
        synapsesInput, previousVisibleToHiddenNeuronsActivation.getSynapsesActivation(), context);

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
        new NeuronsActivation(sample.getActivations().transpose(), 
            NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET));
    UndirectedSynapsesContext context = layerContext.createSynapsesContext(0);
    UndirectedSynapsesActivation visibleNeuronsSynapseActivation = synapses.pushRightToLeft(
        synapsesInput, previousVisibleToHiddenNeuronsActivation.getSynapsesActivation(), context);

    return new RestrictedBoltzmannLayerActivationImpl(visibleNeuronsSynapseActivation,
        new NeuronsActivation(
            visibleNeuronsSynapseActivation.getOutput().getActivations().transpose(), 
            NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) ,
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
   * @param data The training data.
   * @param visibleNeurons The visible neurons.
   * @param hiddenNeurons The hidden neurons.
   * @param learningRate The learning rate.
   * @param matrixFactory The matrix factory.
   * @return The initial connection weights.
   */
  public static Matrix generateInitialConnectionWeights(NeuronsActivation data, 
      Neurons visibleNeurons,
      Neurons hiddenNeurons, double learningRate, MatrixFactory matrixFactory) {

    int initialHiddenUnitBiasWeight = -4;
    Matrix thetas = matrixFactory.createRandn(visibleNeurons.getNeuronCountIncludingBias(),
        hiddenNeurons.getNeuronCountIncludingBias()).mul(learningRate);
    for (int i = 1; i < thetas.getColumns(); i++) {
      thetas.put(0, i, initialHiddenUnitBiasWeight);
    }
    for (int i = 1; i < thetas.getRows(); i++) {
      double[] proportionsOfOnUnits = getProportionsOfOnUnits(data.getActivations());
      double proportionOfTimeUnitActivated = proportionsOfOnUnits[i - 1];
      // Needed to add the following to limit p here, otherwise the log blows up
      proportionOfTimeUnitActivated = Math.max(proportionOfTimeUnitActivated, 0.001);
      double initialVisibleUnitBiasWeight =
          Math.log(proportionOfTimeUnitActivated / (1 - proportionOfTimeUnitActivated));
      thetas.put(i, 0, initialVisibleUnitBiasWeight);
    }
    thetas.put(0, 0, 0);
    return thetas;
  }

  private static double[] getProportionsOfOnUnits(Matrix data) {
    int[] counts = new int[data.getColumns()];
    for (int row = 0; row < data.getRows(); row++) {
      double[] dat = data.getRow(row).toArray();
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

}
