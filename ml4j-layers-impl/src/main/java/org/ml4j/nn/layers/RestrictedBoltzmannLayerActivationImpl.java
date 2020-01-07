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

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.synapses.UndirectedSynapsesActivation;

public class RestrictedBoltzmannLayerActivationImpl implements RestrictedBoltzmannLayerActivation {

  private UndirectedSynapsesActivation synapsesActivation;
  private NeuronsActivation visibleNeuronProbablities;
  private NeuronsActivation hiddenNeuronProbablities;

  /**
   * @param synapsesActivation The activation of the Synapses of this Layer.
   * @param visibleNeuronProbablities The visible neuron probabilities.
   * @param hiddenNeuronProbablities The hidden neuron probablities.
   */
  public RestrictedBoltzmannLayerActivationImpl(UndirectedSynapsesActivation synapsesActivation,
      NeuronsActivation visibleNeuronProbablities, NeuronsActivation hiddenNeuronProbablities) {
    this.synapsesActivation = synapsesActivation;
    this.visibleNeuronProbablities = visibleNeuronProbablities;
    this.hiddenNeuronProbablities = hiddenNeuronProbablities;
  }

  @Override
  public NeuronsActivation getHiddenActivationBinarySample(MatrixFactory matrixFactory) {
    return getBinarySample(hiddenNeuronProbablities, matrixFactory);
  }

  @Override
  public NeuronsActivation getHiddenActivationProbabilities() {
    return hiddenNeuronProbablities;
  }

  @Override
  public UndirectedSynapsesActivation getSynapsesActivation() {
    return synapsesActivation;
  }

  @Override
  public NeuronsActivation getVisibleActivationBinarySample(MatrixFactory matrixFactory) {
    return getBinarySample(visibleNeuronProbablities, matrixFactory);
  }

  @Override
  public NeuronsActivation getVisibleActivationProbablities() {
    return visibleNeuronProbablities;
  }

  private NeuronsActivation getBinarySample(NeuronsActivation probablities,
      MatrixFactory matrixFactory) {
    Matrix rand = matrixFactory.createRand(probablities.getRows(),
        probablities.getColumns());
    Matrix res = probablities.getActivations(matrixFactory).sub(rand);
    EditableMatrix sample = matrixFactory.createMatrix(probablities.getRows(),
        probablities.getColumns()).asEditableMatrix();
    for (int r = 0; r < sample.getRows(); r++) {
      for (int c = 0; c < sample.getColumns(); c++) {

        if (res.get(r, c) > 0) {
          sample.put(r, c, 1);
        }
      }
    }
    return new NeuronsActivationImpl(sample, probablities.getFeatureOrientation());
  }
}
