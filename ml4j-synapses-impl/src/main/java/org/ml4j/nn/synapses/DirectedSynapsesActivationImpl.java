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

package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesActivationImpl extends DirectedSynapsesActivationBase {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DirectedSynapsesActivationImpl.class);

  /**
   * 
   * @param synapses The synapses.
   * @param inputActivation The input activation.
   * @param axonsActivation The axons activation.
   * @param activationFunctionActivation The activation function activation.
   * @param outputActivation The output activation.
   */
  public DirectedSynapsesActivationImpl(DirectedSynapses<?, ?> synapses,
      NeuronsActivation inputActivation, AxonsActivation axonsActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation) {
    super(synapses, inputActivation, axonsActivation, activationFunctionActivation,
        outputActivation);
  }


  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesGradient da,
      DirectedSynapsesContext context, double regularisationLamdba) {

    Axons<?, ?, ?> axons = synapses.getAxons();

    LOGGER.debug("Back propagating through synapses activation....");

    if (axons.getRightNeurons().hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }

    if (axonsActivation == null) {
      throw new IllegalStateException(
          "The synapses activation is expected to contain an AxonsActivation");
    }
    
    NeuronsActivation dz = 
        activationFunctionActivation.backPropagate(da, context).getOutput();
    
    LOGGER.debug("Pushing data right to left through axons...");

    // Will contain bias unit if Axons have left bias unit
    NeuronsActivation inputGradient =
        axons.pushRightToLeft(dz, axonsActivation, context.createAxonsContext()).getOutput();


    Matrix totalTrainableAxonsGradient = null;

    if (axons.isTrainable(context.createAxonsContext())) {

      LOGGER.debug("Calculating Axons Gradients");

      totalTrainableAxonsGradient =
          dz.getActivations()
          .mmul(axonsActivation
              .getPostDropoutInputWithPossibleBias().getActivationsWithBias());

      if (regularisationLamdba != 0) {

        LOGGER.debug("Calculating total regularisation Gradients");

        Matrix connectionWeightsCopy = axons.getDetachedConnectionWeights();

        Matrix firstRow = totalTrainableAxonsGradient.getRow(0);
        Matrix firstColumn = totalTrainableAxonsGradient.getColumn(0);

        totalTrainableAxonsGradient =
            totalTrainableAxonsGradient.addi(connectionWeightsCopy.muli(regularisationLamdba));

        if (axons.getLeftNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putRow(0, firstRow);
        }
        if (axons.getRightNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putColumn(0, firstColumn);
        }
      }
    }

    return new DirectedSynapsesGradientImpl(inputGradient, totalTrainableAxonsGradient);
  }
  
  @Override
  public DirectedSynapsesGradient backPropagate(CostFunctionGradient da,
      DirectedSynapsesContext context, double regularisationLamdba) {

    Axons<?, ?, ?> axons = synapses.getAxons();

    LOGGER.debug("Back propagating through synapses activation....");

    if (axons.getRightNeurons().hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }

    if (axonsActivation == null) {
      throw new IllegalStateException(
          "The synapses activation is expected to contain an AxonsActivation");
    }
   
    NeuronsActivation dz = 
        activationFunctionActivation.backPropagate(da, context).getOutput();
    
    LOGGER.debug("Pushing data right to left through axons...");

    // Will contain bias unit if Axons have left bias unit
    NeuronsActivation inputGradient =
        axons.pushRightToLeft(dz, axonsActivation, context.createAxonsContext()).getOutput();
    
    Matrix totalTrainableAxonsGradient = null;

    if (axons.isTrainable(context.createAxonsContext())) {

      LOGGER.debug("Calculating Axons Gradients");

      totalTrainableAxonsGradient =
          dz.getActivations()
          .mmul(axonsActivation
              .getPostDropoutInputWithPossibleBias().getActivationsWithBias());

      if (regularisationLamdba != 0) {

        LOGGER.debug("Calculating total regularisation Gradients");

        Matrix connectionWeightsCopy = axons.getDetachedConnectionWeights();

        Matrix firstRow = totalTrainableAxonsGradient.getRow(0);
        Matrix firstColumn = totalTrainableAxonsGradient.getColumn(0);

        totalTrainableAxonsGradient =
            totalTrainableAxonsGradient.addi(connectionWeightsCopy.muli(regularisationLamdba));

        if (axons.getLeftNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putRow(0, firstRow);
        }
        if (axons.getRightNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradient.putColumn(0, firstColumn);
        }
      }
    }

    return new DirectedSynapsesGradientImpl(inputGradient, totalTrainableAxonsGradient);
  }
  
 
}
