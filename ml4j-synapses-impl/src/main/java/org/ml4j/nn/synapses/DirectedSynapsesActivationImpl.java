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

package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesActivationImpl implements DirectedSynapsesActivation {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesActivationImpl.class);
  
  private NeuronsActivation inputActivation;
  private AxonsActivation axonsActivation;
  private NeuronsActivation outputActivation;
  private DirectedSynapses<?> synapses;
  
  /**
   * Construct a new default DirectedSynapsesActivation
   * 
   * @param synapses The DirectedSynapses
   * @param inputActivation The input NeuronsActivation of the DirectedSynapses
   *        following a forward propagation
   * @param axonsActivation The axons NeuronsActivation of the DirectedSynapses
   *        following a forward propagation    
   * @param outputActivation The output NeuronsActivation of the DirectedSynapses
   *        following a forward propagation.
   */
  public DirectedSynapsesActivationImpl(DirectedSynapses<?> synapses, 
      NeuronsActivation inputActivation, 
      AxonsActivation axonsActivation, NeuronsActivation outputActivation) {
    this.inputActivation = inputActivation;
    this.outputActivation = outputActivation;
    this.synapses = synapses;
    this.axonsActivation = axonsActivation;
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivation;
  }

  @Override
  public DirectedSynapsesGradient backPropagate(NeuronsActivation da,
      DirectedSynapsesContext context, boolean outerMostSynapses) {
   
    LOGGER.debug(context.toString() + " Back propagating through synapses activation....");
   
    if (synapses.getAxons().getRightNeurons().hasBiasUnit()
        && !da.isBiasUnitIncluded()) {
      LOGGER.debug("Adding zeros for biases to back propagated deltas");
      da = da.withBiasUnit(true, context);
      da.getActivations().putRow(0,
          context.getMatrixFactory().createZeros(1, 
              da.getActivations().getColumns()));
    }
    
    NeuronsActivation axonsOutputActivation = axonsActivation.getOutput();
    
    Matrix dz = outerMostSynapses ? da.getActivations()
        : da.getActivations().mul(synapses.getActivationFunction()
            .activationGradient(axonsOutputActivation, context).getActivations());

    if (da.getFeatureCountIncludingBias() != synapses.getAxons().getRightNeurons()
        .getNeuronCountIncludingBias()) {
      throw new IllegalArgumentException("Expected feature count to be:"
          + synapses.getAxons().getRightNeurons().getNeuronCountIncludingBias() + " but was:"
          + da.getFeatureCountIncludingBias());
    }
    
    NeuronsActivation dzN = new NeuronsActivation(dz, 
        synapses.getAxons().getRightNeurons().hasBiasUnit(),
        da.getFeatureOrientation());

    LOGGER.debug(context.toString() + " Pushing data right to left through axons...");
    NeuronsActivation inputGradient =
        synapses.getAxons().pushRightToLeft(dzN, axonsActivation, 
            context.createAxonsContext()).getOutput();
    
    double numberOfTrainingExamples = da.getActivations().getColumns();
    
    Matrix trainableAxonsGradient = null;
    
    if (synapses.getAxons() instanceof TrainableAxons) {
      trainableAxonsGradient = 
          dz.mmul(this.inputActivation.getActivations()).div(numberOfTrainingExamples);
    }
  
    if (inputGradient.isBiasUnitIncluded()) {
      LOGGER.debug("Removing biases from back propagated deltas");
      inputGradient = new NeuronsActivation(adjustDeltas(inputGradient.getActivations()), false,
          inputGradient.getFeatureOrientation());
    }
    
    return new DirectedSynapsesGradientImpl(inputGradient, 
        trainableAxonsGradient);
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
