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
      DirectedSynapsesContext context, boolean outerMostSynapses, double regularisationLamdba) {
   
    LOGGER.debug("Back propagating through synapses activation....");
    
    if (da.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Back propagated deltas must not contain bias unit");
    }
    
    
    if (synapses.getAxons().getRightNeurons().hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }
    
    NeuronsActivation axonsOutputActivation = axonsActivation.getOutput();
    
    Matrix activationGradient = synapses.getActivationFunction()
        .activationGradient(axonsOutputActivation.withBiasUnit(false, context), context)
        .getActivations();  
  
    Matrix dz = outerMostSynapses ? da.getActivations()
        : da.getActivations().mul(activationGradient);

    if (da.getFeatureCountIncludingBias() != synapses.getAxons().getRightNeurons()
        .getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Expected feature count to be:"
          + synapses.getAxons().getRightNeurons().getNeuronCountExcludingBias() + " but was:"
          + da.getFeatureCountIncludingBias());
    }
    
    // Does not contain output bias unit
    NeuronsActivation dzN = new NeuronsActivation(dz, 
        false,
        da.getFeatureOrientation());

    LOGGER.debug("Pushing data right to left through axons...");
    
    // Will contain bias unit if Axons have left  bias unit
    NeuronsActivation inputGradient =
        synapses.getAxons().pushRightToLeft(dzN, axonsActivation, 
            context.createAxonsContext()).getOutput();
    
     
    if (inputGradient.isBiasUnitIncluded()) {
      LOGGER.debug("Removing biases from back propagated deltas");
      inputGradient = new NeuronsActivation(inputGradient.getActivations(), 
          inputGradient.isBiasUnitIncluded(),
          inputGradient.getFeatureOrientation()).withBiasUnit(false, context);
    }
         
    Matrix totalTrainableAxonsGradient = null;
    
    if (synapses.getAxons() instanceof TrainableAxons) {
     
      LOGGER.debug("Calculating Axons Gradients");

      totalTrainableAxonsGradient = 
          dz.mmul(this.inputActivation.getActivations());
      
      if (regularisationLamdba != 0) {
       
        LOGGER.debug("Calculating total regularisation Gradients");
   
        Matrix connectionWeightsCopy = synapses.getAxons().getDetachedConnectionWeights();

        Matrix firstRow = totalTrainableAxonsGradient.getRow(0);
        Matrix firstColumn = totalTrainableAxonsGradient.getColumn(0);

        totalTrainableAxonsGradient = totalTrainableAxonsGradient
            .addi(connectionWeightsCopy.muli(regularisationLamdba));
        
        if (synapses.getAxons().getLeftNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putRow(0, firstRow);
        }
        if (synapses.getAxons().getRightNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putColumn(0, firstColumn);
        }
      }
    }

    return new DirectedSynapsesGradientImpl(inputGradient, 
        totalTrainableAxonsGradient);
  }

  @Override
  public DirectedSynapses<?> getSynapses() {
    return synapses;
  }

  @Override
  public double getAverageRegularisationCost(double regularisationLambda) {
    LOGGER.debug("Calculating average regularisation cost");
    return getTotalRegularisationCost(regularisationLambda) 
        / outputActivation.getActivations().getRows();
  }

  @Override
  public double getTotalRegularisationCost(double regularisationLambda) {
  
    if (regularisationLambda != 0) {

      LOGGER.debug("Calculating total regularisation cost");
      
      Matrix weightsWithBiases = synapses.getAxons().getDetachedConnectionWeights();

      int[] rows = new int[weightsWithBiases.getRows()
          - (this.getSynapses().getAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0)];
      int[] cols = new int[weightsWithBiases.getColumns()
          - (this.getSynapses().getAxons().getRightNeurons().hasBiasUnit() ? 1 : 0)];
      for (int j = 0; j < weightsWithBiases.getColumns(); j++) {
        cols[j - (this.getSynapses().getAxons().getRightNeurons().hasBiasUnit() ? 1 : 0)] = j;
      }
      for (int j = 1; j < weightsWithBiases.getRows(); j++) {
        rows[j - (this.getSynapses().getAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0)] = j;
      }

      Matrix weightsWithoutBiases = weightsWithBiases.get(rows, cols);

      double regularisationMatrix = weightsWithoutBiases.mul(weightsWithoutBiases).sum();
      return ((regularisationLambda) * regularisationMatrix) / 2;
    } else {
      return 0;
    }
  }
}
