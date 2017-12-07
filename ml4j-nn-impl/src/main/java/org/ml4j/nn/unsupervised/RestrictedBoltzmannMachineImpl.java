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

package org.ml4j.nn.unsupervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

import java.util.Arrays;
import java.util.List;

public class RestrictedBoltzmannMachineImpl implements RestrictedBoltzmannMachine {

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
    throw new UnsupportedOperationException("Not yet implemented");
  }
}
