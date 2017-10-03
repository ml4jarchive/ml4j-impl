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

package org.ml4j.nn.unsupervised.mocks;

import org.ml4j.mocks.MatrixMock;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.axons.mocks.AxonsMock;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.mocks.ForwardPropagationMock;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.unsupervised.AutoEncoder;
import org.ml4j.nn.unsupervised.AutoEncoderContext;
import org.ml4j.util.SerializationHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple mock implementation of AutoEncoder consisting of 2 mock FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class AutoEncoderMock implements AutoEncoder {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(AutoEncoderMock.class);
  
  private FeedForwardLayer<?, ?> encodingLayer;
  private FeedForwardLayer<?, ?> decodingLayer;
  
  public AutoEncoderMock(FeedForwardLayer<?, ?> encodingLayer, 
      FeedForwardLayer<?, ?> decodingLayer) {
    this.encodingLayer = encodingLayer;
    this.decodingLayer = decodingLayer;
  }
 
  @Override
  public void train(NeuronsActivation trainingDataActivations, AutoEncoderContext trainingContext) {
    LOGGER.debug("Mock training AutoEncoderMock - no op for now");
    
    AxonsMock encodingLayerAxons = (AxonsMock)encodingLayer.getPrimaryAxons();
    AxonsMock decodingLayerAxons = (AxonsMock)decodingLayer.getPrimaryAxons();
    SerializationHelper helper = new SerializationHelper("/Users/michael/Desktop/sers");
    double[][] layer1Array = helper.deserialize(double[][].class, "layer1A");
    double[][] layer2Array = helper.deserialize(double[][].class, "layer2A");
    encodingLayerAxons.setConnectionWeights(new MatrixMock(layer1Array));
    decodingLayerAxons.setConnectionWeights(new MatrixMock(layer2Array));

  }

  @Override
  public List<FeedForwardLayer<?, ?>> getLayers() {
    List<FeedForwardLayer<?, ?>> layers = new ArrayList<>();
    layers.add(encodingLayer);
    layers.add(decodingLayer);
    return layers;
  }

  @Override
  public int getNumberOfLayers() {
    return 2;
  }

  @Override
  public FeedForwardLayer<?, ?> getLayer(int layerIndex) {
    return getLayers().get(layerIndex);
  }

  @Override
  public FeedForwardLayer<?, ?> getFirstLayer() {
    return encodingLayer;
  }

  @Override
  public FeedForwardLayer<?, ?> getFinalLayer() {
    return decodingLayer;
  }

  @Override
  public AutoEncoder dup() {
    return new AutoEncoderMock(encodingLayer.dup(), decodingLayer.dup());
  }

  @Override
  public NeuronsActivation encode(NeuronsActivation unencoded, AutoEncoderContext context) {
    LOGGER.debug("Encoding through AutoEncoderMock");
    if (context.getEndLayerIndex() == null
        || context.getEndLayerIndex() >= (this.getNumberOfLayers() - 1)) {
      throw new IllegalArgumentException("End layer index for encoding through AutoEncoder "
          + " must be specified and must not be the index of the last layer");
    }
    return forwardPropagate(unencoded, context).getOutputs();
  }

  @Override
  public NeuronsActivation decode(NeuronsActivation encoded, AutoEncoderContext context) {
    LOGGER.debug("Decoding through AutoEncoderMock");
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
