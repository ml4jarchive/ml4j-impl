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

import org.ml4j.mocks.MatrixMock;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.axons.AxonsImpl;

import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.mocks.ForwardPropagationMock;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.unsupervised.mocks.AutoEncoderMock;
import org.ml4j.util.SerializationHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Default implementation of AutoEncoder consisting of 2 default FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class AutoEncoderImpl implements AutoEncoder {

  private static final Logger LOGGER = LoggerFactory.getLogger(AutoEncoderImpl.class);
  
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
  public AutoEncoderImpl(FeedForwardLayer<?, ?> encodingLayer,
      FeedForwardLayer<?, ?> decodingLayer) {
    this.layers = new ArrayList<FeedForwardLayer<?, ?>>();
    this.layers.add(encodingLayer);
    this.layers.add(decodingLayer);
  }

  @Override
  public void train(NeuronsActivation trainingDataActivations, AutoEncoderContext trainingContext) {
    LOGGER.debug(
        "Mock training AutoEncoderMock - simulating training by loading pre-trained weights");

    AxonsImpl encodingLayerAxons = (AxonsImpl) getLayer(0).getPrimaryAxons();
    AxonsImpl decodingLayerAxons = (AxonsImpl) getLayer(1).getPrimaryAxons();
    SerializationHelper helper =
        new SerializationHelper(AutoEncoderMock.class.getClassLoader(), "pretrainedweights");
    double[][] layer1Array = helper.deserialize(double[][].class, "layer1");
    double[][] layer2Array = helper.deserialize(double[][].class, "layer2");
    encodingLayerAxons.setConnectionWeights(new MatrixMock(layer1Array));
    decodingLayerAxons.setConnectionWeights(new MatrixMock(layer2Array));
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
  public AutoEncoder dup() {
    return new AutoEncoderImpl(getLayer(0), getLayer(1));
  }

  @Override
  public NeuronsActivation encode(NeuronsActivation unencoded, AutoEncoderContext context) {
    LOGGER.debug("Encoding through AutoEncoder");
    if (context.getEndLayerIndex() == null
        || context.getEndLayerIndex() >= (this.getNumberOfLayers() - 1)) {
      throw new IllegalArgumentException("End layer index for encoding through AutoEncoder "
          + " must be specified and must not be the index of the last layer");
    }
    return forwardPropagate(unencoded, context).getOutputs();
  }

  @Override
  public NeuronsActivation decode(NeuronsActivation encoded, AutoEncoderContext context) {
    LOGGER.debug("Decoding through AutoEncoder");
    if (context.getStartLayerIndex() == 0) {
      throw new IllegalArgumentException("Start layer index for decoding through AutoEncoder "
          + " must not be 0 - the index of the first layer");
    }
    return forwardPropagate(encoded, context).getOutputs();
  }

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      AutoEncoderContext context) {
    
    int endLayerIndex =
        context.getEndLayerIndex() == null ? (getNumberOfLayers() - 1) : context.getEndLayerIndex();

    LOGGER.debug("Forward propagating through AutoEncoderMock from layerIndex:"
        + context.getStartLayerIndex() + " to layerIndex:" + endLayerIndex);
        
    NeuronsActivation inFlightActivations = inputActivation;
    int layerIndex = 0;

    for (FeedForwardLayer<?, ?> layer : getLayers()) {

      if (layerIndex >= context.getStartLayerIndex() && layerIndex <= endLayerIndex) {

        inFlightActivations =
            layer.forwardPropagate(inFlightActivations, context.createLayerContext(layerIndex))
                .getOutput();
      }
      layerIndex++;

    }
    
    return new ForwardPropagationMock(inFlightActivations);
  }
}
