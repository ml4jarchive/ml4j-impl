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

package org.ml4j.nn.unsupervised;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerActivation;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class RestrictedBoltzmannMachineImpl implements RestrictedBoltzmannMachine {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(RestrictedBoltzmannMachineImpl.class);

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private RestrictedBoltzmannLayer<FullyConnectedAxons> restrictedBoltzmannLayer;

  public RestrictedBoltzmannMachineImpl(
      RestrictedBoltzmannLayer<FullyConnectedAxons> restrictedBoltzmannLayer) {
    this.restrictedBoltzmannLayer = restrictedBoltzmannLayer;
  }

  /**
   * @param axons The Axons.
   * @param visibleActivationFunction The visible ActivationFunction
   * @param hiddenActivationFunction The hidden ActivationFunction
   */
  public RestrictedBoltzmannMachineImpl(FullyConnectedAxons axons,
      ActivationFunction visibleActivationFunction, ActivationFunction hiddenActivationFunction) {
    this.restrictedBoltzmannLayer = new RestrictedBoltzmannLayerImpl(axons,
        visibleActivationFunction, hiddenActivationFunction);
  }

  /**
   * @param visibleNeurons The visible Neurons.
   * @param hiddenNeurons The hidden Neurons.
   * @param visibleActivationFunction The visible ActivationFunction.
   * @param hiddenActivationFunction The hidden ActivationFunction.
   * @param matrixFactory The MatrixFactory.
   */
  public RestrictedBoltzmannMachineImpl(Neurons visibleNeurons, Neurons hiddenNeurons,
      ActivationFunction visibleActivationFunction, ActivationFunction hiddenActivationFunction,
      MatrixFactory matrixFactory) {
    this.restrictedBoltzmannLayer = new RestrictedBoltzmannLayerImpl(visibleNeurons, hiddenNeurons,
        visibleActivationFunction, hiddenActivationFunction, matrixFactory);
  }

  @Override
  public RestrictedBoltzmannMachine dup() {
    return new RestrictedBoltzmannMachineImpl(restrictedBoltzmannLayer.dup());
  }

  @Override
  public RestrictedBoltzmannLayer<FullyConnectedAxons> getFinalLayer() {
    return restrictedBoltzmannLayer;
  }

  @Override
  public RestrictedBoltzmannLayer<FullyConnectedAxons> getFirstLayer() {
    return restrictedBoltzmannLayer;
  }

  @Override
  public RestrictedBoltzmannLayer<FullyConnectedAxons> getLayer(int layerIndex) {
    if (layerIndex == 0) {
      return restrictedBoltzmannLayer;
    } else {
      throw new IllegalArgumentException(
          "RestrictedBoltzmannMachines have only a single Layer available at index 0");
    }
  }

  @Override
  public List<RestrictedBoltzmannLayer<FullyConnectedAxons>> getLayers() {
    return Arrays.asList(restrictedBoltzmannLayer);
  }

  @Override
  public int getNumberOfLayers() {
    return 1;
  }

  @Override
  public void train(NeuronsActivation trainingActivations,
      RestrictedBoltzmannMachineContext trainingContext) {

    final int numberOfEpochs = trainingContext.getTrainingEpochs();

    LOGGER.info("Training the RestrictedBoltzmannMachine for " + numberOfEpochs + " epochs");

    NeuronsActivation data = null;
    NeuronsActivation lastReconstructions = null;

    for (int i = 0; i < numberOfEpochs; i++) {

      if (trainingContext.getTrainingMiniBatchSize() == null) {

        data = trainingActivations;

        lastReconstructions = trainOnBatch(data, trainingContext);

        LOGGER.info("Epoch:" + (i + 1) + " Average Reconstruction Error:"
            + getAverageReconstructionError(trainingActivations, lastReconstructions));

      } else {
        int miniBatchSize = trainingContext.getTrainingMiniBatchSize();
        int numberOfTrainingElements = trainingActivations.getActivations().getRows();
        int numberOfBatches = (numberOfTrainingElements - 1) / miniBatchSize + 1;
        for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++) {
          int startRowIndex = batchIndex * miniBatchSize;
          int endRowIndex =
              Math.min(startRowIndex + miniBatchSize - 1, numberOfTrainingElements - 1);
          int[] rowIndexes = new int[endRowIndex - startRowIndex + 1];
          for (int r = startRowIndex; r <= endRowIndex; r++) {
            rowIndexes[r - startRowIndex] = r;
          }

          Matrix dataBatch = trainingActivations.getActivations().getRows(rowIndexes);

          NeuronsActivation batchDataActivations =
              new NeuronsActivation(dataBatch, trainingActivations.getFeatureOrientation());
          data = batchDataActivations;

          lastReconstructions = trainOnBatch(data, trainingContext);

          LOGGER.trace("Epoch:" + i + " batch " + batchIndex + " Average Reconstruction Error:"
              + getAverageReconstructionError(batchDataActivations, lastReconstructions));

          batchIndex++;

        }
        LOGGER.info("Epoch:" + i + " Average Reconstruction Error:"
            + getAverageReconstructionError(data, lastReconstructions));
      }
    }
  }


  private NeuronsActivation trainOnBatch(NeuronsActivation data,
      RestrictedBoltzmannMachineContext trainingContext) {

    NeuronsActivation visibleActivations = data;

    RestrictedBoltzmannSamplingActivation samplingActivation =
        performGibbsSampling(data, 1, trainingContext);

    RestrictedBoltzmannLayerActivation firstHiddenNeuronsDataActivation =
        samplingActivation.getFirstHiddenNeuronsActivation();
    RestrictedBoltzmannLayerActivation firstVisibleNeuronsReconstructionLayerActivation =
        samplingActivation.getFirstVisibleNeuronsReconstructionLayerActivation();

    RestrictedBoltzmannLayerActivation lastVisibleNeuronsReconstructionLayerActivation =
        samplingActivation.getLastVisibleNeuronsReconstructionLayerActivation();

    // Perform up to 1 gibbs sampling steps
    for (int c = 0; c < 1; c++) {

      // Push the visible data to the hidden neurons

      RestrictedBoltzmannLayerActivation hiddenNeuronsDataActivation =
          getFirstLayer().activateHiddenNeuronsFromVisibleNeuronsData(visibleActivations,
              trainingContext.getLayerContext(0));

      if (firstHiddenNeuronsDataActivation == null) {
        firstHiddenNeuronsDataActivation = hiddenNeuronsDataActivation;
      }

      // Push a hidden neuron sample to the visible neurons to get a reconstruction

      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation =
          getFirstLayer().activateVisibleNeuronsFromHiddenNeuronsSample(
              firstHiddenNeuronsDataActivation, trainingContext.getLayerContext(0));

      lastVisibleNeuronsReconstructionLayerActivation = visibleNeuronsReconstructionLayerActivation;

      if (firstVisibleNeuronsReconstructionLayerActivation == null) {
        firstVisibleNeuronsReconstructionLayerActivation =
            visibleNeuronsReconstructionLayerActivation;
      }

      visibleActivations =
          visibleNeuronsReconstructionLayerActivation.getVisibleActivationProbablities();

    }

    NeuronsActivation lastReconstructions =
        lastVisibleNeuronsReconstructionLayerActivation.getVisibleActivationProbablities();

    RestrictedBoltzmannLayerActivation contrastiveDivergenceActivation =
        getFirstLayer().activateHiddenNeuronsFromVisibleNeuronsReconstruction(
            lastVisibleNeuronsReconstructionLayerActivation, trainingContext.getLayerContext(0));

    // Calculate the statistics and the weight adjustment
    Matrix adjustment = getWeightsAdjustment(firstHiddenNeuronsDataActivation,
        firstVisibleNeuronsReconstructionLayerActivation, contrastiveDivergenceActivation,
        trainingContext);

    getFirstLayer().getPrimaryAxons().adjustConnectionWeights(adjustment,
        ConnectionWeightsAdjustmentDirection.ADDITION);

    return lastReconstructions;

  }

  private Matrix getWeightsAdjustment(
      RestrictedBoltzmannLayerActivation hiddenNeuronsDataActivation,
      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation,
      RestrictedBoltzmannLayerActivation contrastiveDivergenceActivation,
      RestrictedBoltzmannMachineContext trainingContext) {

    // Calculate positive statisics

    NeuronsActivationWithPossibleBiasUnit visibleAxonsDataActivations = hiddenNeuronsDataActivation
        .getSynapsesActivation().getAxonsActivation().getPostDropoutInputWithPossibleBias();

    NeuronsActivationWithPossibleBiasUnit hiddenAxonsDataActivationsTransposed =
        visibleNeuronsReconstructionLayerActivation.getSynapsesActivation().getAxonsActivation()
            .getPostDropoutInputWithPossibleBias();

    NeuronsActivationWithPossibleBiasUnit hiddenAxonsDataActivations =
        new NeuronsActivationWithPossibleBiasUnit(
            hiddenAxonsDataActivationsTransposed.getActivations().transpose(),
            getFirstLayer().getHiddenNeurons().hasBiasUnit(),
            NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, false);

    Matrix positiveStatistics = getContrastiveDivergenceStatistics(visibleAxonsDataActivations,
        hiddenAxonsDataActivations, trainingContext);

    // Calculate negative statistics

    // Push the reconstruction to the hidden neurons to get an activation we can use to
    // calculate
    // contrastive divergence statisitics.

    NeuronsActivationWithPossibleBiasUnit visibleAxonsReconstructionActivations =
        contrastiveDivergenceActivation.getSynapsesActivation().getAxonsActivation()
            .getPostDropoutInputWithPossibleBias();

    NeuronsActivation hiddenAxonsReconstructionActivationsWithoutBias =
        contrastiveDivergenceActivation.getHiddenActivationProbabilities();

    NeuronsActivationWithPossibleBiasUnit hiddenAxonsReconstructionActivations =
        new NeuronsActivationWithPossibleBiasUnit(
            hiddenAxonsReconstructionActivationsWithoutBias.getActivations(), false,
            NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, false).withBiasUnit(true,
                trainingContext);

    Matrix negativeStatistics =
        getContrastiveDivergenceStatistics(visibleAxonsReconstructionActivations,
            hiddenAxonsReconstructionActivations, trainingContext);

    Matrix adjustment =
        positiveStatistics.sub(negativeStatistics).mul(trainingContext.getTrainingLearningRate());

    return adjustment;

  }

  private Matrix getContrastiveDivergenceStatistics(
      NeuronsActivationWithPossibleBiasUnit visibleNeuronsActivation,
      NeuronsActivationWithPossibleBiasUnit hiddenNeuronsActivation,
      RestrictedBoltzmannMachineContext trainingContext) {

    if (visibleNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException(
          "Visible neurons activation must be columns span feature" + " set");
    }

    if (hiddenNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException(
          "Hidden neurons activation must be columns span feature" + " set");
    }

    return getAveragePairwiseRowProducts(visibleNeuronsActivation.getActivations(),
        hiddenNeuronsActivation.getActivations(), trainingContext.getMatrixFactory());
  }


  private Matrix getAveragePairwiseRowProducts(Matrix matrix1, Matrix matrix2,
      MatrixFactory matrixFactory) {
    Matrix result = matrixFactory.createMatrix(matrix1.getColumns(), matrix2.getColumns());

    for (int i = 0; i < matrix1.getRows(); i++) {
      Matrix vector1 = matrix1.getRow(i);
      Matrix vector2 = matrix2.getRow(i);

      result.addi(getPairwiseVectorProduct(vector1, vector2, matrixFactory));
    }
    return result.div(matrix1.getRows());

  }

  private Matrix getPairwiseVectorProduct(Matrix vector1, Matrix vector2,
      MatrixFactory matrixFactory) {
    Matrix result = matrixFactory.createMatrix(vector1.getColumns(), vector2.getColumns());
    for (int i = 0; i < vector1.getColumns(); i++) {
      for (int j = 0; j < vector2.getColumns(); j++) {
        result.put(i, j, vector1.get(0, i) * vector2.get(0, j));
      }
    }
    return result;
  }

  private double getAverageReconstructionError(NeuronsActivation data,
      NeuronsActivation reconstructions) {
    Matrix diff = data.getActivations().sub(reconstructions.getActivations());
    return diff.mul(diff).sum() / data.getActivations().getRows();
  }

  @Override
  public RestrictedBoltzmannSamplingActivation performGibbsSampling(
      NeuronsActivation initialVisibleActivations, int cdn,
      RestrictedBoltzmannMachineContext context) {

    RestrictedBoltzmannLayerActivation lastVisibleNeuronsReconstructionLayerActivation = null;
    RestrictedBoltzmannLayerActivation firstHiddenNeuronsDataActivation = null;
    RestrictedBoltzmannLayerActivation firstVisibleNeuronsReconstructionLayerActivation = null;

    NeuronsActivation visibleActivations = initialVisibleActivations;

    // Perform up to 1 gibbs sampling steps
    for (int c = 0; c < cdn; c++) {

      // Push the visible data to the hidden neurons

      RestrictedBoltzmannLayerActivation hiddenNeuronsDataActivation =
          getFirstLayer().activateHiddenNeuronsFromVisibleNeuronsData(visibleActivations,
              context.getLayerContext(0));

      if (firstHiddenNeuronsDataActivation == null) {
        firstHiddenNeuronsDataActivation = hiddenNeuronsDataActivation;
      }

      // Push a hidden neuron sample to the visible neurons to get a reconstruction

      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation =
          getFirstLayer().activateVisibleNeuronsFromHiddenNeuronsSample(
              firstHiddenNeuronsDataActivation, context.getLayerContext(0));

      lastVisibleNeuronsReconstructionLayerActivation = visibleNeuronsReconstructionLayerActivation;

      if (firstVisibleNeuronsReconstructionLayerActivation == null) {
        firstVisibleNeuronsReconstructionLayerActivation =
            visibleNeuronsReconstructionLayerActivation;
      }

      visibleActivations =
          visibleNeuronsReconstructionLayerActivation.getVisibleActivationProbablities();

    }

    return new RestrictedBoltzmannSamplingActivationImpl(firstHiddenNeuronsDataActivation,
        firstVisibleNeuronsReconstructionLayerActivation,
        lastVisibleNeuronsReconstructionLayerActivation);

  }
}
