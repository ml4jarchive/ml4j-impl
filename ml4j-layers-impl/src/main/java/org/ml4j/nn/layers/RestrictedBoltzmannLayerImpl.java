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

package org.ml4j.nn.layers;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.UndirectedSynapses;
import org.ml4j.nn.synapses.UndirectedSynapsesImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class RestrictedBoltzmannLayerImpl implements RestrictedBoltzmannLayer<FullyConnectedAxons> {

  private static final Logger LOGGER = LoggerFactory.getLogger(RestrictedBoltzmannLayerImpl.class);

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private FullyConnectedAxons axons;

  private UndirectedSynapses<?, ?> synapses;

  /**
   * @param axons The Axons.
   * @param visibleActivationFunction The visible ActivationFunction
   * @param hiddenActivationFunction The hidden ActivationFunction
   */
  public RestrictedBoltzmannLayerImpl(FullyConnectedAxons axons,
      ActivationFunction visibleActivationFunction, ActivationFunction hiddenActivationFunction) {
    this.axons = axons;
    this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
        hiddenActivationFunction);
  }

  /**
   * @param visibleNeurons The visible Neurons.
   * @param hiddenNeurons The hidden Neurons.
   * @param visibleActivationFunction The visible ActivationFunction.
   * @param hiddenActivationFunction The hidden ActivationFunction.
   * @param matrixFactory The MatrixFactory.
   */
  public RestrictedBoltzmannLayerImpl(Neurons visibleNeurons, Neurons hiddenNeurons,
      ActivationFunction visibleActivationFunction, ActivationFunction hiddenActivationFunction,
      MatrixFactory matrixFactory) {
    this.axons = new FullyConnectedAxonsImpl(visibleNeurons, hiddenNeurons, matrixFactory);
    this.synapses = new UndirectedSynapsesImpl<Neurons, Neurons>(axons, visibleActivationFunction,
        hiddenActivationFunction);
  }

  @Override
  public RestrictedBoltzmannLayer<FullyConnectedAxons> dup() {
    return new RestrictedBoltzmannLayerImpl(axons.dup(), synapses.getLeftActivationFunction(),
        synapses.getRightActivationFunction());
  }

  @Override
  public FullyConnectedAxons getPrimaryAxons() {
    return axons;
  }

  @Override
  public List<UndirectedSynapses<?, ?>> getSynapses() {
    return Arrays.asList(synapses);
  }

  @Override
  public NeuronsActivation getOptimalVisibleActivationsForHiddenNeuron(int hiddenNeuronIndex,
      UndirectedLayerContext undirectedLayerContext) {
    LOGGER.debug("Obtaining optimal input for hidden neuron with index:" + hiddenNeuronIndex);
    Matrix weights = getPrimaryAxons().getDetachedConnectionWeights();
    int countJ = weights.getRows() - (getPrimaryAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0);
    double[] maximisingInputFeatures = new double[countJ];
    boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

    for (int j = 0; j < countJ; j++) {
      double wij = getWij(j, hiddenNeuronIndex, weights, hasBiasUnit);
      double sum = 0;

      if (wij != 0) {

        for (int j2 = 0; j2 < countJ; j2++) {
          double weight = getWij(j2, hiddenNeuronIndex, weights, hasBiasUnit);
          if (weight != 0) {
            sum = sum + Math.pow(weight, 2);
          }
        }
        sum = Math.sqrt(sum);
      }
      maximisingInputFeatures[j] = wij / sum;
    }
    return new NeuronsActivation(
        undirectedLayerContext.getMatrixFactory()
            .createMatrix(new double[][] {maximisingInputFeatures}),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  private double getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
    int indICorrected = indI + (hasBiasUnit ? 1 : 0);
    return weights.get(indICorrected, indJ);
  }
}
