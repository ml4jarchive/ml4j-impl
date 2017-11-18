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

package org.ml4j.nn;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.CrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.MultiClassCrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.SumSquaredErrorCostFunction;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerGradient;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Default base implementation of a FeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public abstract class FeedForwardNeuralNetworkBase<C extends FeedForwardNeuralNetworkContext, 
    N extends FeedForwardNeuralNetwork<C,N>> 
    implements FeedForwardNeuralNetwork<C, N> {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardNeuralNetworkBase.class);
  
  private List<FeedForwardLayer<?, ?>> layers;
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  /**
   * Constructor for a multi-layer FeedForwardNeuralNetwork.
   * 
   * @param layers The layers
   */
  public FeedForwardNeuralNetworkBase(FeedForwardLayer<?, ?>... layers) {
    this.layers = new ArrayList<FeedForwardLayer<?, ?>>();
    this.layers.addAll(Arrays.asList(layers));
  }

  protected void train(NeuronsActivation trainingDataActivations, 
      NeuronsActivation trainingLabelActivations, C trainingContext) {

    final int numberOfTrainingIterations = trainingContext.getTrainingIterations();
    
    LOGGER.info("Training the FeedForwardNeuralNetwork for "
          + numberOfTrainingIterations + " iterations");
    
    // Perform the addition of the bias once here for efficiency - without this logic the
    // bias would be added on each iteration.
    if (getFirstLayer().getSynapses()
        .get(0).getAxons().getLeftNeurons().hasBiasUnit()
        && !trainingDataActivations.isBiasUnitIncluded()) {
      trainingDataActivations = trainingDataActivations.withBiasUnit(true, trainingContext);
    }
    
    for (int i = 0; i < numberOfTrainingIterations; i++) {

      CostAndGradients costAndGradients = getCostAndGradients(trainingDataActivations, 
          trainingLabelActivations, trainingContext);
      
      LOGGER.info("Iteration:" + i + " Cost:" + costAndGradients.getAverageCost());
      
      adjustConnectionWeights(trainingContext, 
          costAndGradients.getAverageTrainableAxonsGradients());
    }
  }
 
  protected CostAndGradients getCostAndGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutputActivations, C trainingContext) {
   
    List<DirectedSynapses<?>> finalLayerSynapses = getFinalLayer()
        .getSynapses();

    DirectedSynapses<?> finalSynapses = finalLayerSynapses.get(finalLayerSynapses.size() - 1);

    boolean expectBiasUnitForOutput = finalSynapses.getAxons().getRightNeurons().hasBiasUnit();

    if (expectBiasUnitForOutput && !desiredOutputActivations.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Expected desired output activations to be with bias");
    } else if (!expectBiasUnitForOutput && desiredOutputActivations.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Expected desired output activations to be without bias");
    } 
    if (expectBiasUnitForOutput && !desiredOutputActivations.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Expected desired output activations to be with bias");
    } else if (!expectBiasUnitForOutput && desiredOutputActivations.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Expected desired output activations to be without bias");
    } 
    
    // Desired output activations should be without bias.
    if (desiredOutputActivations.isBiasUnitIncluded()) {
      desiredOutputActivations = desiredOutputActivations.withBiasUnit(false, trainingContext);
    }
    
    final CostFunction costFunction = getCostFunction(trainingContext.getMatrixFactory());
    
    // Forward propagate the trainingDataActivations through the entire AutoEncoder
    ForwardPropagation forwardPropagation =
        forwardPropagate(inputActivations, trainingContext);
    
    // When using either of the cross entropy cross functions, 
    // the deltas we backpropagate
    // end up being the difference between the target activations ( which are the 
    // same as the trainingDataActivations as this is an AutoEncoder), and the
    // activations resulting from the forward propagation
    Matrix deltasM = forwardPropagation.getOutputs().getActivations()
        .sub(desiredOutputActivations.getActivations());
    
    // The deltas we back propagate are in the transposed orientation to the inputs
    NeuronsActivation deltas = new NeuronsActivation(deltasM.transpose(),
        forwardPropagation.getOutputs().isBiasUnitIncluded(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
   
    // Back propagate the deltas through the nework
    BackPropagation backPropagation = forwardPropagation.backPropagate(deltas, trainingContext);

    // Obtain the gradients of each set of Axons we wish to train - for this example it is
    // all the Axons
    List<Matrix> totalTrainableAxonsGradients = new ArrayList<>();
    List<DirectedLayerGradient> reversed = new ArrayList<>();
    reversed.addAll(backPropagation.getDirectedLayerGradients());
    Collections.reverse(reversed);

    for (DirectedLayerGradient gradient : reversed) {
      for (DirectedSynapsesGradient synapsesGradient : gradient.getSynapsesGradients()) {
        
        Matrix totalTrainableAxonsGradient = synapsesGradient.getTotalTrainableAxonsGradient();
        
        if (totalTrainableAxonsGradient != null) {
          totalTrainableAxonsGradients.add(totalTrainableAxonsGradient);
        }
      }
    }
    
    // Obtain the cost from the cost function
    LOGGER.debug("Calculating total cost function cost");
    double totalCost = costFunction.getTotalCost(desiredOutputActivations.getActivations(),
        forwardPropagation.getOutputs().getActivations());
    
    double totalRegularisationCost = forwardPropagation.getTotalRegularisationCost(trainingContext);
        
    double totalCostWithRegularisation = totalCost + totalRegularisationCost;
    
    Collections.reverse(totalTrainableAxonsGradients);

    int numberOfTrainingExamples = inputActivations.getActivations().getRows();
    
    return new CostAndGradients(totalCostWithRegularisation, 
          totalTrainableAxonsGradients, numberOfTrainingExamples);
    
  }
  
  private List<TrainableAxons<?, ?, ?>> getTrainableAxonsList() {
    
    List<TrainableAxons<?, ?, ?>> trainableAxonsList = new ArrayList<>();
    for (int layerIndex = 0; layerIndex < getNumberOfLayers(); layerIndex++) {

      FeedForwardLayer<?, ?> layer = getLayer(layerIndex);
      for (DirectedSynapses<?> synapses : layer.getSynapses()) {
        if (synapses.getAxons() instanceof TrainableAxons) {
          TrainableAxons<?, ?, ?> axons = (TrainableAxons<?, ?, ?>) synapses.getAxons();
          trainableAxonsList.add(axons);
        }
      }
    }
    return trainableAxonsList;
    
  }
  
  private void adjustConnectionWeights(C trainingContext, List<Matrix> trainableAxonsGradients) {
    double learningRate = trainingContext.getTrainingLearningRate();
    
    List<TrainableAxons<?, ?, ?>> trainableAxonsList = getTrainableAxonsList();
    
    for (int axonsIndex = 0; axonsIndex < trainableAxonsGradients.size(); axonsIndex++) {
      TrainableAxons<?, ?, ?> trainableAxons = trainableAxonsList.get(axonsIndex);
      // Transpose the axon gradients into matrices that correspond to the orientation of the
      // connection weights ( COLUMNS_SPAN_FEATURE_SET )
      Matrix axonsGrad = trainableAxonsGradients.get(axonsIndex).transpose();

      // Adjust the weights of each set of Axons by subtracting the learning-rate scaled
      // gradient matrices
      trainableAxons.adjustConnectionWeights(axonsGrad.mul(learningRate), 
          ConnectionWeightsAdjustmentDirection.SUBTRACTION);
    }
  }

  @Override
  public List<FeedForwardLayer<?, ?>> getLayers() {
    return layers;
  }

  @Override
  public int getNumberOfLayers() {
    return layers.size();
  }

  @Override
  public FeedForwardLayer<?, ?> getLayer(int layerIndex) {
    return layers.get(layerIndex);
  }

  @Override
  public FeedForwardLayer<?, ?> getFirstLayer() {
    return layers.get(0);
  }

  @Override
  public FeedForwardLayer<?, ?> getFinalLayer() {
    return layers.get(getNumberOfLayers() - 1);

  }

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      FeedForwardNeuralNetworkContext context) {
    
    int endLayerIndex =
        context.getEndLayerIndex() == null ? (getNumberOfLayers() - 1) : context.getEndLayerIndex();

    LOGGER.debug("Forward propagating through FeedForwardNeuralNetwork from layerIndex:"
        + context.getStartLayerIndex() + " to layerIndex:" + endLayerIndex);
        
    NeuronsActivation inFlightActivations = inputActivation;
    int layerIndex = 0;
    List<DirectedLayerActivation> activations = new ArrayList<>();
    for (FeedForwardLayer<?, ?> layer : getLayers()) {

      if (layerIndex >= context.getStartLayerIndex() && layerIndex <= endLayerIndex) {

        DirectedLayerActivation inFlightLayerActivations = 
            layer.forwardPropagate(inFlightActivations, context.getLayerContext(layerIndex));
        activations.add(inFlightLayerActivations);
        inFlightActivations = inFlightLayerActivations.getOutput();
      }
      layerIndex++;
    }
    
    return new ForwardPropagationImpl(activations, inFlightActivations);
  }
  
  /**
   * @return The default cost function for use by this Network.
   */
  protected CostFunction getCostFunction(MatrixFactory matrixFactory) {

    List<DirectedSynapses<?>> synapseList = getFinalLayer().getSynapses();
    DirectedSynapses<?> finalSynapses = synapseList.get(synapseList.size() - 1);
    DifferentiableActivationFunction activationFunction = finalSynapses.getActivationFunction();
    if (activationFunction instanceof SigmoidActivationFunction) {
      LOGGER.debug("Defaulting to use CrossEntropyCostFunction");
      return new CrossEntropyCostFunction();
    } else if (activationFunction instanceof SoftmaxActivationFunction) {
      LOGGER.debug("Defaulting to use MultiClassCrossEntropyCostFunction");
      return new MultiClassCrossEntropyCostFunction();
    } else if (activationFunction instanceof LinearActivationFunction) {
      LOGGER.debug("Defaulting to use SumSquredErrorCostFunction");
      return new SumSquaredErrorCostFunction();
    } else {
      throw new UnsupportedOperationException(
          "Default cost function not yet defined for:" + activationFunction.getClass());
    }
  }
}
