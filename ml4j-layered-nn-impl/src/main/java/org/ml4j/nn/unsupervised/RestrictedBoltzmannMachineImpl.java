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

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.AxonWeightsAdjustment;
import org.ml4j.nn.axons.AxonWeightsAdjustmentDirection;
import org.ml4j.nn.axons.AxonWeightsAdjustmentImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerActivation;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RestrictedBoltzmannMachineImpl implements RestrictedBoltzmannMachine {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(RestrictedBoltzmannMachineImpl.class);

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> restrictedBoltzmannLayer;

  public RestrictedBoltzmannMachineImpl(
      RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> restrictedBoltzmannLayer) {
    this.restrictedBoltzmannLayer = restrictedBoltzmannLayer;
  }

  /**
   * @param axons The Axons.
   * @param visibleActivationFunction The visible ActivationFunction
   * @param hiddenActivationFunction The hidden ActivationFunction
   */
  public RestrictedBoltzmannMachineImpl(TrainableAxons<?, ?, ?> axons,
      ActivationFunction<?, ?> visibleActivationFunction, 
      ActivationFunction<?, ?> hiddenActivationFunction) {
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
  public RestrictedBoltzmannMachineImpl(AxonsFactory axonsFactory, Neurons visibleNeurons, Neurons hiddenNeurons,
      ActivationFunction<?, ?> visibleActivationFunction, 
      ActivationFunction<?, ?> hiddenActivationFunction,
      MatrixFactory matrixFactory) {
    this.restrictedBoltzmannLayer = new RestrictedBoltzmannLayerImpl(axonsFactory, visibleNeurons, hiddenNeurons,
        visibleActivationFunction, hiddenActivationFunction, matrixFactory);
  }

  @Override
  public RestrictedBoltzmannMachine dup() {
    return new RestrictedBoltzmannMachineImpl(restrictedBoltzmannLayer.dup());
  }

  /*
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

	*/

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
            + getAverageReconstructionError(trainingContext.getMatrixFactory(),trainingActivations, lastReconstructions));

      } else {
        int miniBatchSize = trainingContext.getTrainingMiniBatchSize();
        int numberOfTrainingElements = trainingActivations.getActivations(trainingContext.getMatrixFactory()).getColumns();
        int numberOfBatches = (numberOfTrainingElements - 1) / miniBatchSize + 1;
        for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++) {
          int startColumnIndex = batchIndex * miniBatchSize;
          int endColumnIndex =
              Math.min(startColumnIndex + miniBatchSize - 1, numberOfTrainingElements - 1);
          int[] colIndexes = new int[endColumnIndex - startColumnIndex + 1];
          for (int c = startColumnIndex; c <= endColumnIndex; c++) {
        	  colIndexes[c - startColumnIndex] = c;
          }

          Matrix dataBatch = trainingActivations.getActivations(trainingContext.getMatrixFactory()).getColumns(colIndexes);

          NeuronsActivation batchDataActivations =
              new NeuronsActivationImpl(dataBatch, trainingActivations.getFeatureOrientation());
          data = batchDataActivations;
          
          lastReconstructions = trainOnBatch(data, trainingContext);

          LOGGER.trace("Epoch:" + i + " batch " + batchIndex + " Average Reconstruction Error:"
              + getAverageReconstructionError(trainingContext.getMatrixFactory(), batchDataActivations, lastReconstructions));

          batchIndex++;

        }
        LOGGER.info("Epoch:" + i + " Average Reconstruction Error:"
            + getAverageReconstructionError(trainingContext.getMatrixFactory(), data, lastReconstructions));
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
          restrictedBoltzmannLayer.activateHiddenNeuronsFromVisibleNeuronsData(visibleActivations,
              trainingContext.getLayerContext());

      if (firstHiddenNeuronsDataActivation == null) {
        firstHiddenNeuronsDataActivation = hiddenNeuronsDataActivation;
      }

      // Push a hidden neuron sample to the visible neurons to get a reconstruction

      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation =
    		  restrictedBoltzmannLayer.activateVisibleNeuronsFromHiddenNeuronsSample(
              firstHiddenNeuronsDataActivation, trainingContext.getLayerContext());

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
    		restrictedBoltzmannLayer.activateHiddenNeuronsFromVisibleNeuronsReconstruction(
            lastVisibleNeuronsReconstructionLayerActivation, trainingContext.getLayerContext());

    // Calculate the statistics and the weight adjustment
    Matrix adjustment = getWeightsAdjustment(firstHiddenNeuronsDataActivation,
        firstVisibleNeuronsReconstructionLayerActivation, contrastiveDivergenceActivation,
        trainingContext);
        
    int[] rows = new int[adjustment.getRows() - 1];
    for (int r = 0; r < rows.length; r++) {
    	rows[r] = r + 1;
    }
    
    int[] columns = new int[adjustment.getColumns() - 1];
    for (int c = 0; c < columns.length; c++) {
    	columns[c] = c + 1;
    }
    
    Matrix weightAdjustment = adjustment.get(rows, columns);
    
    Matrix leftToRightBiases = adjustment.getRow(0).getColumns(columns);
    Matrix rightToLeftBiases = adjustment.getColumn(0).getRows(rows);
    
    AxonWeightsAdjustment axonWeightsAdjustment = new AxonWeightsAdjustmentImpl(weightAdjustment.transpose(), leftToRightBiases.transpose(), rightToLeftBiases);

    restrictedBoltzmannLayer.getPrimaryAxons().adjustAxonWeights(axonWeightsAdjustment,
        AxonWeightsAdjustmentDirection.ADDITION);

    return lastReconstructions;

  }

  private Matrix getWeightsAdjustment(
      RestrictedBoltzmannLayerActivation hiddenNeuronsDataActivation,
      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation,
      RestrictedBoltzmannLayerActivation contrastiveDivergenceActivation,
      RestrictedBoltzmannMachineContext trainingContext) {

    // Calculate positive statisics
	  
    NeuronsActivation visibleAxonsDataActivationsWithoutBiasActivation = hiddenNeuronsDataActivation
        .getSynapsesActivation().getAxonsActivation().getPostDropoutInput();
    
    NeuronsActivation hiddenAxonsDataActivationsWithoutBiasTransposed =
        visibleNeuronsReconstructionLayerActivation.getSynapsesActivation().getAxonsActivation()
            .getPostDropoutInput();

    NeuronsActivationWithPossibleBiasUnit hiddenAxonsDataActivationsWithoutBias =
        new NeuronsActivationWithPossibleBiasUnit(
        		hiddenAxonsDataActivationsWithoutBiasTransposed.getActivations(trainingContext.getMatrixFactory()).asEditableMatrix(),
            false,
            NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
    
    NeuronsActivationWithPossibleBiasUnit hiddenAxonsDataActivations = hiddenAxonsDataActivationsWithoutBias.withBiasUnit(
    		restrictedBoltzmannLayer.getHiddenNeurons().hasBiasUnit(), trainingContext);
    
    NeuronsActivationWithPossibleBiasUnit visibleAxonsDataActivationsWithoutBias =
            new NeuronsActivationWithPossibleBiasUnit(
            		visibleAxonsDataActivationsWithoutBiasActivation.getActivations(trainingContext.getMatrixFactory()).asEditableMatrix(),
                false,
                NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
    
    
    NeuronsActivationWithPossibleBiasUnit visibleAxonsDataActivations = visibleAxonsDataActivationsWithoutBias.withBiasUnit(
    		restrictedBoltzmannLayer.getVisibleNeurons().hasBiasUnit(), trainingContext);

    Matrix positiveStatistics = getContrastiveDivergenceStatistics(visibleAxonsDataActivations,
        hiddenAxonsDataActivations, trainingContext);

    // Calculate negative statistics

    // Push the reconstruction to the hidden neurons to get an activation we can use to
    // calculate
    // contrastive divergence statisitics.

    NeuronsActivation visibleAxonsReconstructionActivationsWithoutBias =
        contrastiveDivergenceActivation.getSynapsesActivation().getAxonsActivation()
            .getPostDropoutInput();

    NeuronsActivation hiddenAxonsReconstructionActivationsWithoutBias =
        contrastiveDivergenceActivation.getHiddenActivationProbabilities();

    NeuronsActivationWithPossibleBiasUnit hiddenAxonsReconstructionActivations =
        new NeuronsActivationWithPossibleBiasUnit(
            hiddenAxonsReconstructionActivationsWithoutBias.getActivations(trainingContext.getMatrixFactory()).asEditableMatrix(), false,
            NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false).withBiasUnit(restrictedBoltzmannLayer.getHiddenNeurons().hasBiasUnit(),
                trainingContext);
    
    NeuronsActivationWithPossibleBiasUnit visibleAxonsReconstructionActivations =
            new NeuronsActivationWithPossibleBiasUnit(
            		visibleAxonsReconstructionActivationsWithoutBias.getActivations(trainingContext.getMatrixFactory()).asEditableMatrix(), false,
                NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false).withBiasUnit(restrictedBoltzmannLayer.getVisibleNeurons().hasBiasUnit(),
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
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException(
          "Visible neurons activation must be columns span feature" + " set");
    }

    if (hiddenNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException(
          "Hidden neurons activation must be columns span feature" + " set");
    }

    return getAveragePairwiseColumnProducts(visibleNeuronsActivation.getActivations(),
            hiddenNeuronsActivation.getActivations(), trainingContext.getMatrixFactory());
  }

  private Matrix getAveragePairwiseColumnProducts(Matrix matrix1, Matrix matrix2,
	      MatrixFactory matrixFactory) {
	    EditableMatrix result = matrixFactory.createMatrix(matrix1.getRows(), matrix2.getRows()).asEditableMatrix();

	    for (int i = 0; i < matrix1.getColumns(); i++) {
	      Matrix vector1 = matrix1.getColumn(i);
	      Matrix vector2 = matrix2.getColumn(i);

	      result.addi(getPairwiseVectorProduct(vector1, vector2, matrixFactory));
	    }
	    return result.div(matrix1.getRows());

	  }

	  private Matrix getPairwiseVectorProduct(Matrix vector1, Matrix vector2,
	      MatrixFactory matrixFactory) {
	    EditableMatrix result = matrixFactory.createMatrix(vector1.getRows(), vector2.getRows()).asEditableMatrix();
	    for (int i = 0; i < vector1.getRows(); i++) {
	      for (int j = 0; j < vector2.getRows(); j++) {
	        result.put(i, j, vector1.get(i, 0) * vector2.get(j, 0));
	      }
	    }
	    return result;
	  }

  private double getAverageReconstructionError(MatrixFactory matrixFactory, NeuronsActivation data,
      NeuronsActivation reconstructions) {
	  
    Matrix diff = data.getActivations(matrixFactory).sub(reconstructions.getActivations(matrixFactory));
    return diff.mul(diff).sum() / data.getActivations(matrixFactory).getColumns();
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
    		  restrictedBoltzmannLayer.activateHiddenNeuronsFromVisibleNeuronsData(visibleActivations,
              context.getLayerContext());

      if (firstHiddenNeuronsDataActivation == null) {
        firstHiddenNeuronsDataActivation = hiddenNeuronsDataActivation;
      }

      // Push a hidden neuron sample to the visible neurons to get a reconstruction

      RestrictedBoltzmannLayerActivation visibleNeuronsReconstructionLayerActivation =
    		  restrictedBoltzmannLayer.activateVisibleNeuronsFromHiddenNeuronsSample(
              firstHiddenNeuronsDataActivation, context.getLayerContext());

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

  @Override
  public AutoEncoder createAutoEncoder() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public NeuronsActivation decodeToBinary(NeuronsActivation hiddenUnitActivatons,
      RestrictedBoltzmannMachineContext context) {
    return restrictedBoltzmannLayer
        .activateVisibleNeuronsFromHiddenNeurons(hiddenUnitActivatons, context.getLayerContext())
        .getVisibleActivationBinarySample(context.getMatrixFactory());
  }

  @Override
  public NeuronsActivation decodeToProbabilities(NeuronsActivation hiddenUnitActivatons,
      RestrictedBoltzmannMachineContext context) {
    return restrictedBoltzmannLayer
        .activateVisibleNeuronsFromHiddenNeurons(hiddenUnitActivatons, context.getLayerContext())
        .getVisibleActivationProbablities();
  }

  @Override
  public NeuronsActivation encodeToBinary(NeuronsActivation visibleUnitActivations,
      RestrictedBoltzmannMachineContext context) {
    return restrictedBoltzmannLayer
        .activateHiddenNeuronsFromVisibleNeuronsData(
            visibleUnitActivations, context.getLayerContext())
        .getHiddenActivationBinarySample(context.getMatrixFactory());
  }

@Override
public RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> getLayer() {
	return restrictedBoltzmannLayer;
}
}
