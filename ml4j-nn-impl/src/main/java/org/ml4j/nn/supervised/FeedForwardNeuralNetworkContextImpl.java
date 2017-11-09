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

package org.ml4j.nn.supervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.DirectedLayerContextImpl;

import java.util.HashMap;
import java.util.Map;

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

  private int startLayerIndex;
  
  private Integer endLayerIndex;
 
  private Integer trainingIterations;
  
  private Double trainingLearningRate;
  
  private Map<Integer, DirectedLayerContext> directedLayerContexts;
  
  /**
   * Construct a default AutoEncoderContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   */
  public FeedForwardNeuralNetworkContextImpl(MatrixFactory matrixFactory, 
      int startLayerIndex, Integer endLayerIndex) {
    this.matrixFactory = matrixFactory;
    this.startLayerIndex = startLayerIndex;
    this.endLayerIndex = endLayerIndex;
    if (endLayerIndex != null && startLayerIndex > endLayerIndex) {
      throw new IllegalArgumentException("Start layer index cannot be greater "
          + "than end layer index");
    }
    this.directedLayerContexts = new HashMap<>();
  }
  
  @Override
  public MatrixFactory getMatrixFactory() {
    return matrixFactory;
  }

  @Override
  public DirectedLayerContext getLayerContext(int layerIndex) {
    
    DirectedLayerContext layerContext = directedLayerContexts.get(layerIndex);
    if (layerContext == null) {
      layerContext = new DirectedLayerContextImpl(matrixFactory);
      directedLayerContexts.put(layerIndex, layerContext);
    }
    
    return layerContext;
  }

  @Override
  public int getStartLayerIndex() {
    return startLayerIndex;
  }

  @Override
  public Integer getEndLayerIndex() {
    return endLayerIndex;
  }

  @Override
  public int getTrainingIterations() {
    if (trainingIterations == null) {
      throw new IllegalStateException("Number of training iterations not set on context");
    }
    return trainingIterations.intValue();
  }

  @Override
  public double getTrainingLearningRate() {
    if (trainingLearningRate == null) {
      throw new IllegalStateException("Training learning rate not set on context");
    }
    return trainingLearningRate.doubleValue();
  }

  @Override
  public void setTrainingIterations(int trainingIterations) {
    this.trainingIterations = trainingIterations;
  }

  @Override
  public void setTrainingLearningRate(double trainingLearningRate) {
    this.trainingLearningRate = trainingLearningRate;
  }
}
