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

import org.ml4j.Matrix;
import org.ml4j.nn.BackPropagation;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.ForwardPropagationImpl;
import org.ml4j.nn.axons.AxonsImpl;
import org.ml4j.nn.costfunctions.MultiClassCrossEntropyCostFunction;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerGradient;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;

import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Default implementation of AutoEncoder consisting of 2 default FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class SupervisedFeedForwardNeuralNetworkImpl 
    implements SupervisedFeedForwardNeuralNetwork {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(SupervisedFeedForwardNeuralNetworkImpl.class);
  
  private List<FeedForwardLayer<?, ?>> layers;
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  /**
   * Constructor for a simple 2-layer AutoEncoder.
   * 
   * @param encodingLayer The encoding Layer
   * @param decodingLayer The decoding Layer
   */
  public SupervisedFeedForwardNeuralNetworkImpl(FeedForwardLayer<?, ?> encodingLayer,
      FeedForwardLayer<?, ?> decodingLayer) {
    this.layers = new ArrayList<FeedForwardLayer<?, ?>>();
    this.layers.add(encodingLayer);
    this.layers.add(decodingLayer);
  }
  
  /**
   * Constructor for a multi-layer AutoEncoder.
   * 
   * @param layers The layers
   */
  public SupervisedFeedForwardNeuralNetworkImpl(FeedForwardLayer<?, ?>... layers) {
    this.layers = new ArrayList<FeedForwardLayer<?, ?>>();
    this.layers.addAll(Arrays.asList(layers));
  }

  @Override
  public void train(NeuronsActivation trainingDataActivations, 
      NeuronsActivation trainingLabelActivations, FeedForwardNeuralNetworkContext trainingContext) {

    final int numberOfTrainingIterations = trainingContext.getTrainingIterations();
    
    LOGGER.info("Training SupervisedFeedForwardNeuralNetworkImpl...");

    LOGGER.debug("Initialising Axon weights...");
    List<AxonsImpl> axonsList = new ArrayList<>();
    for (int layerIndex = 0; layerIndex < getNumberOfLayers(); layerIndex++) {

      FeedForwardLayer<?, ?> layer = getLayer(layerIndex);
      for (DirectedSynapses<?> synapses : layer.getSynapses()) {
        AxonsImpl axons = (AxonsImpl) synapses.getAxons();

        Matrix weights = trainingContext.getMatrixFactory().createRandn(
            axons.getLeftNeurons().getNeuronCountIncludingBias(),
            axons.getRightNeurons().getNeuronCountIncludingBias());

        axons.setConnectionWeights(weights.mul(0.01));
        axonsList.add(axons);
      }
    }
    
    for (int i = 0; i < numberOfTrainingIterations; i++) {

      // Forward propagate the trainingDataActivations through the entire AutoEncoder
      ForwardPropagation forwardPropagation =
          forwardPropagate(trainingDataActivations, trainingContext);
      
      // We're going to use the cross entropy cross function to train the AutoEncoder
      final MultiClassCrossEntropyCostFunction costFunction 
          = new MultiClassCrossEntropyCostFunction();
      
      // When using the cross entropy cross function, the deltas we backpropagate
      // end up being the difference between the target activations ( which are the 
      // same as the trainingDataActivations as this is an AutoEncoder), and the
      // activations resulting from the forward propagation
      Matrix deltasM = forwardPropagation.getOutputs().getActivations()
          .sub(trainingLabelActivations.getActivations());
      
      // The deltas we back propagate are in the transposed orientation to the inputs
      NeuronsActivation deltas = new NeuronsActivation(deltasM.transpose(),
          forwardPropagation.getOutputs().isBiasUnitIncluded(),
          NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
     
      // Back propagate the deltas through the nework
      BackPropagation backPropagation = forwardPropagation.backPropagate(deltas, trainingContext);

      // Obtain the gradients of each set of Axons we wish to train - for this example it is
      // all the Axons
      List<Matrix> axonsGradients = new ArrayList<>();
      List<DirectedLayerGradient> reversed = new ArrayList<>();
      reversed.addAll(backPropagation.getDirectedLayerGradients());
      Collections.reverse(reversed);

      for (DirectedLayerGradient gradient : reversed) {
        for (DirectedSynapsesGradient synapsesGradient : gradient.getSynapsesGradients()) {
          axonsGradients.add(synapsesGradient.getAxonsGradient());
        }
      }
      
      Collections.reverse(axonsGradients);

      
      double learningRate = trainingContext.getTrainingLearningRate();
      
      for (int axonsIndex = 0; axonsIndex < axonsGradients.size(); axonsIndex++) {
        AxonsImpl axons = axonsList.get(axonsIndex);
        // Transpose the axon gradients into matrices that correspond to the orientation of the
        // connection weights ( COLUMNS_SPAN_FEATURE_SET )
        Matrix axonsGrad = axonsGradients.get(axonsIndex).transpose();
        Matrix exisitingAxonsWeights = axons.getConnectionWeights();

        // Adjust the weights of each set of Axons by subtracting the learning-rate scaled
        // gradient matrices
        Matrix newAxonsWeights = exisitingAxonsWeights.sub(axonsGrad.mul(learningRate));
        axons.setConnectionWeights(newAxonsWeights);
      }

      // Obtain the cost from the cost function
      double cost = costFunction.getCost(trainingLabelActivations.getActivations(),
          forwardPropagation.getOutputs().getActivations());
      LOGGER.info("COST:" + cost);
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
  public SupervisedFeedForwardNeuralNetwork dup() {
    return new SupervisedFeedForwardNeuralNetworkImpl(getLayer(0), getLayer(1));
  }

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      FeedForwardNeuralNetworkContext context) {
    
    int endLayerIndex =
        context.getEndLayerIndex() == null ? (getNumberOfLayers() - 1) : context.getEndLayerIndex();

    LOGGER.debug("Forward propagating through SupervisedFeedForwardNeuralNetwork from layerIndex:"
        + context.getStartLayerIndex() + " to layerIndex:" + endLayerIndex);
        
    NeuronsActivation inFlightActivations = inputActivation;
    int layerIndex = 0;
    List<DirectedLayerActivation> activations = new ArrayList<>();
    for (FeedForwardLayer<?, ?> layer : getLayers()) {

      if (layerIndex >= context.getStartLayerIndex() && layerIndex <= endLayerIndex) {

        DirectedLayerActivation inFlightLayerActivations = 
            layer.forwardPropagate(inFlightActivations, context.createLayerContext(layerIndex));
        activations.add(inFlightLayerActivations);
        inFlightActivations = inFlightLayerActivations.getOutput();
      }
      layerIndex++;
    }
    
    return new ForwardPropagationImpl(activations, inFlightActivations);
  }
}
