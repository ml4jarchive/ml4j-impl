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

package org.ml4j.nn.synapses.mocks;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesActivationMock implements DirectedSynapsesActivation {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesActivationMock.class);
  
  private NeuronsActivation outputActivation;
  private NeuronsActivation inputActivation;
  private DirectedSynapses<?> synapses;
  
  /**
   * Construct a new mock DirectedSynapsesActivation
   * 
   * @param synapses The DirectedSynapses
   * @param inputActivation The input NeuronsActivation of the DirectedSynapses
   *        following a forward propagation
   * @param outputActivation The output NeuronsActivation of the DirectedSynapses
   *        following a forward propagation.
   */
  public DirectedSynapsesActivationMock(DirectedSynapses<?> synapses, 
      NeuronsActivation inputActivation, 
      NeuronsActivation outputActivation) {
    this.inputActivation = inputActivation;
    this.outputActivation = outputActivation;
    this.synapses = synapses;
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivation;
  }

  @Override
  public DirectedSynapsesGradient backPropagate(NeuronsActivation outerActivationGradient,
      DirectedSynapsesContext context, boolean outerLayer) {
    LOGGER.debug(context.toString() + " Back propagating through synapses activation....");
    
    NeuronsActivation input = outerActivationGradient;

    if (synapses.getAxons().getRightNeurons().hasBiasUnit()
        && !outerActivationGradient.isBiasUnitIncluded()) {
      LOGGER.debug("Adding zeros for biases to back propagated deltas");
      input = outerActivationGradient.withBiasUnit(true, context);
      input.getActivations().putRow(0,
          context.getMatrixFactory().createZeros(1, 
              input.getActivations().getColumns()));

    }

    NeuronsActivation activationFunctionGradient = outerLayer ? outputActivation
        : synapses.getActivationFunction().activationGradient(outputActivation, context);

    Matrix dz =
        input.getActivations().mul(
            activationFunctionGradient.getActivations().transpose());
    
    if (input.getFeatureCountIncludingBias() != synapses.getAxons().getRightNeurons()
        .getNeuronCountIncludingBias()) {
      throw new IllegalArgumentException("Expected feature count to be:"
          + synapses.getAxons().getRightNeurons().getNeuronCountIncludingBias() + " but was:"
          + input.getFeatureCountIncludingBias());
    }
    
    NeuronsActivation dzN = new NeuronsActivation(dz, 
        synapses.getAxons().getRightNeurons().hasBiasUnit(),
        outerActivationGradient.getFeatureOrientation());

    LOGGER.debug(context.toString() + " Pushing data right to left through axons...");
    NeuronsActivation inputGradient =
        synapses.getAxons().pushRightToLeft(dzN, null, context.createAxonsContext()).getOutput();
    
    Matrix axonsGradient = dz
        .mmul(this.inputActivation.getActivations());

    if (inputGradient.isBiasUnitIncluded()) {
      LOGGER.debug("Removing biases from back propagated deltas");
      inputGradient = new NeuronsActivation(adjustDeltas(inputGradient.getActivations()), false,
          inputGradient.getFeatureOrientation());
    }
    
    return new DirectedSynapsesGradientMock(inputGradient, 
        axonsGradient);
  }
  
  private Matrix adjustDeltas(Matrix deltas) {

    int[] cols = new int[deltas.getColumns()];
    int[] rows = new int[deltas.getRows() - 1];
    for (int j = 0; j < deltas.getColumns(); j++) {
      cols[j] = j;
    }
    for (int j = 1; j < deltas.getRows(); j++) {
      rows[j - 1] = j;
    }
    deltas = deltas.get(rows, cols);

    return deltas;
  }

  @Override
  public DirectedSynapses<?> getSynapses() {
    return synapses;
  }

}
