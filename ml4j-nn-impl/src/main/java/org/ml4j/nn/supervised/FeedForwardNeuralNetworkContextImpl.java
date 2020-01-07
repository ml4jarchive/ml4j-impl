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

import org.ml4j.MatrixFactory;
import org.ml4j.nn.BackPropagationListener;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagationListener;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.optimisation.GradientDescentOptimisationStrategy;
import org.ml4j.nn.optimisation.TrainingLearningRateAdjustmentStrategy;

/**
 * Simple default implementation of FeedForwardNeuralNetworkContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class FeedForwardNeuralNetworkContextImpl implements FeedForwardNeuralNetworkContext {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  /**
   * The MatrixFactory we configure for this context.
   */
  private MatrixFactory matrixFactory;

  //private int startLayerIndex;

  //private Integer endLayerIndex;
  
  private DirectedComponentsContext directedComponentsContext;

  private Integer trainingEpochs;

  private Float trainingLearningRate;

  private Integer trainingMiniBatchSize;

  private Integer lastTrainingEpochIndex;

  //private Map<Integer, DirectedLayerContext> directedLayerContexts;

  private GradientDescentOptimisationStrategy gradientDescentOptimisationStrategy;

  private TrainingLearningRateAdjustmentStrategy trainingLearningRateAdjustmentStrategy;

  private ForwardPropagationListener forwardPropagationListener;

  private BackPropagationListener backPropagationListener;

  /**
   * Construct a default AutoEncoderContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   */
  public FeedForwardNeuralNetworkContextImpl(MatrixFactory matrixFactory, boolean isTrainingContext) {
    this.matrixFactory = matrixFactory;
    this.directedComponentsContext = new DirectedComponentsContextImpl(matrixFactory, isTrainingContext);
  }

  @Override
  public MatrixFactory getMatrixFactory() {
    return matrixFactory;
  }

  /*
  @Override
  public DirectedLayerContext getLayerContext(int layerIndex) {

    DirectedLayerContext layerContext = directedLayerContexts.get(layerIndex);
    if (layerContext == null) {
      layerContext = new DirectedLayerContextImpl(layerIndex, matrixFactory);
      directedLayerContexts.put(layerIndex, layerContext);
    }

    return layerContext;
  }
  */

  /*
  @Override
  public int getStartLayerIndex() {
    return startLayerIndex;
  }

  @Override
  public Integer getEndLayerIndex() {
    return endLayerIndex;
  }
  */

  @Override
  public int getTrainingEpochs() {
    if (trainingEpochs == null) {
      throw new IllegalStateException("Number of training epochs not set on context");
    }
    return trainingEpochs.intValue();
  }

  @Override
  public float getTrainingLearningRate() {
    if (trainingLearningRate == null) {
      throw new IllegalStateException("Training learning rate not set on context");
    }
    return trainingLearningRate.floatValue();
  }

  @Override
  public void setTrainingEpochs(int trainingEpochs) {
    this.trainingEpochs = trainingEpochs;
  }

  @Override
  public void setTrainingLearningRate(float trainingLearningRate) {
    this.trainingLearningRate = trainingLearningRate;
  }

  @Override
  public Integer getTrainingMiniBatchSize() {
    return trainingMiniBatchSize;
  }

  @Override
  public void setTrainingMiniBatchSize(Integer trainingMiniBatchSize) {
    this.trainingMiniBatchSize = trainingMiniBatchSize;
  }

  @Override
  public GradientDescentOptimisationStrategy getGradientDescentOptimisationStrategy() {
    return gradientDescentOptimisationStrategy;
  }

  @Override
  public void setGradientDescentOptimisationStrategy(
      GradientDescentOptimisationStrategy gradientDescentOptimisationStrategy) {
    this.gradientDescentOptimisationStrategy = gradientDescentOptimisationStrategy;
  }

  @Override
  public Integer getLastTrainingEpochIndex() {
    return lastTrainingEpochIndex;
  }

  @Override
  public TrainingLearningRateAdjustmentStrategy getTrainingLearningRateAdjustmentStrategy() {
    return trainingLearningRateAdjustmentStrategy;
  }

  @Override
  public void setTrainingLearningRateAdjustmentStrategy(
      TrainingLearningRateAdjustmentStrategy trainingLearningRateAdjustmentStrategy) {
    this.trainingLearningRateAdjustmentStrategy = trainingLearningRateAdjustmentStrategy;
  }

  @Override
  public void setLastTrainingEpochIndex(Integer lastTrainingEpochIndex) {
    this.lastTrainingEpochIndex = lastTrainingEpochIndex;
  }

  @Override
  public BackPropagationListener getBackPropagationListener() {
    return backPropagationListener;
  }

  @Override
  public ForwardPropagationListener getForwardPropagationListener() {
    return forwardPropagationListener;
  }

  @Override
  public void setBackPropagationListener(BackPropagationListener backPropagationListener) {
    this.backPropagationListener = backPropagationListener;

  }

  @Override
  public void setForwardPropagationListener(ForwardPropagationListener forwardPropagationListener) {
    this.forwardPropagationListener = forwardPropagationListener;
  }

  @Override
  public DirectedComponentsContext getDirectedComponentsContext() {
	return this.directedComponentsContext;
  }
}
