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

import org.ml4j.nn.FeedForwardNeuralNetworkBase;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of AutoEncoder consisting of FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class AutoEncoderImpl extends 
    FeedForwardNeuralNetworkBase<AutoEncoderContext, AutoEncoder> implements AutoEncoder {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(AutoEncoderImpl.class);
 
  /**
   * Constructor for a simple 2-layer AutoEncoder.
   * 
   * @param encodingLayer The encoding Layer
   * @param decodingLayer The decoding Layer
   */
  public AutoEncoderImpl(FeedForwardLayer<?, ?> encodingLayer,
      FeedForwardLayer<?, ?> decodingLayer) {
      super(encodingLayer, decodingLayer);
  }
  
  /**
   * Constructor for a multi-layer AutoEncoder.
   * 
   * @param layers The layers
   */
  public AutoEncoderImpl(FeedForwardLayer<?, ?>... layers) {
      super(layers);
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
  public void train(NeuronsActivation inputActivations, AutoEncoderContext trainingContext) {
    super.train(inputActivations, inputActivations, trainingContext);
  }
}
