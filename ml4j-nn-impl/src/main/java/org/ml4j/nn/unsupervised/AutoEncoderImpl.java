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

import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;

import java.util.List;

/**
 * Default implementation of AutoEncoder consisting of 2 default FeedForwardLayers.
 *
 * @author Michael Lavelle
 */
public class AutoEncoderImpl implements AutoEncoder {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  public AutoEncoderImpl(FeedForwardLayer<?, ?> encodingLayer,
      FeedForwardLayer<?, ?> decodingLayer) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public void train(NeuronsActivation trainingDataActivations, AutoEncoderContext trainingContext) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public List<FeedForwardLayer<?, ?>> getLayers() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getNumberOfLayers() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public FeedForwardLayer<?, ?> getLayer(int layerIndex) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public FeedForwardLayer<?, ?> getFirstLayer() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public FeedForwardLayer<?, ?> getFinalLayer() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public AutoEncoder dup() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public NeuronsActivation encode(NeuronsActivation unencoded, AutoEncoderContext context) {
    throw new UnsupportedOperationException("Not implemented yet");

  }

  @Override
  public NeuronsActivation decode(NeuronsActivation encoded, AutoEncoderContext context) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      AutoEncoderContext context) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
