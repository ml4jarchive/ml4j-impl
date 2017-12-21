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
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
      DirectedSynapsesInput inputActivation, DirectedDipoleGraph<AxonsActivation> axonsActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation) {
    super(synapses, inputActivation, axonsActivation, activationFunctionActivation,
        outputActivation);
  }


  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesGradient da,
      DirectedSynapsesContext context) {

    LOGGER.debug("Back propagating through synapses activation....");

    validateAxonsAndAxonsActivation();

    NeuronsActivation dz = activationFunctionActivation.backPropagate(da, context).getOutput();

    return backPropagateThroughAxons(dz, context);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(CostFunctionGradient da,
      DirectedSynapsesContext context) {

    LOGGER.debug("Back propagating through synapses activation....");

    validateAxonsAndAxonsActivation();

    NeuronsActivation dz = activationFunctionActivation.backPropagate(da, context).getOutput();

    return backPropagateThroughAxons(dz, context);
  }

  private void validateAxonsAndAxonsActivation() {

    if (synapses.getRightNeurons().hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }

    if (axonsActivationGraph == null || axonsActivationGraph.getParallelPaths().isEmpty()) {
      throw new IllegalStateException(
          "The synapses activation is expected to contain an AxonsActivation path");
    } else {

      boolean allEmpty = true;
      for (DirectedPath<AxonsActivation> parallelPath : axonsActivationGraph.getParallelPaths()) {
        if (parallelPath != null && !parallelPath.getEdges().isEmpty()) {
          allEmpty = false;
        }
      }
      if (allEmpty) {
        throw new IllegalStateException(
            "The synapses activation is expected to contain an AxonsActivation");
      }
    }

  }

  protected DirectedSynapsesGradient backPropagateThroughAxons(NeuronsActivation dz,
      SynapsesContext synapsesContext) {

    LOGGER.debug("Pushing data right to left through axons...");

    NeuronsActivation residualSynapsesInput = inputActivation.getResidualInput();
    if (residualSynapsesInput != null && axonsActivationGraph.getParallelPaths().size() != 2) {
      throw new UnsupportedOperationException("Not supported yet");
    }
    
    List<AxonsGradient> totalTrainableAxonsGradients = new ArrayList<>();

    NeuronsActivation inputGradient = null;
    Matrix totalInputGradientMatrix = null;
    NeuronsActivation totalResidualGradient = null;

    int pathIndex = 0;
    for (DirectedPath<AxonsActivation> parallelAxonsPath : axonsActivationGraph
        .getParallelPaths()) {

      NeuronsActivation gradientToBackPropagate = dz;

      int axonsIndex = parallelAxonsPath.getEdges().size() - 1;

      List<AxonsActivation> reversedAxonsActivations = new ArrayList<AxonsActivation>();
      reversedAxonsActivations.addAll(parallelAxonsPath.getEdges());
      Collections.reverse(reversedAxonsActivations);

      for (AxonsActivation axonsActivation : reversedAxonsActivations) {

        AxonsContext axonsContext = synapsesContext.getAxonsContext(pathIndex, axonsIndex);

        // Will contain bias unit if Axons have left bias unit
        inputGradient = axonsActivation.getAxons()
            .pushRightToLeft(gradientToBackPropagate, axonsActivation, axonsContext).getOutput();

        AxonsGradient totalTrainableAxonsGradient =
            getTrainableAxonsGradient(axonsActivation, axonsContext, dz);
        if (totalTrainableAxonsGradient != null) {
          totalTrainableAxonsGradients.add(totalTrainableAxonsGradient);
        }
        if (totalInputGradientMatrix == null) {
          totalInputGradientMatrix = inputGradient.getActivations();
        } else {
          if (residualSynapsesInput != null && pathIndex == 1) {
            totalInputGradientMatrix = totalInputGradientMatrix.add(inputGradient.getActivations());
          } else {
            totalResidualGradient = inputGradient;
          }
        }


        gradientToBackPropagate = inputGradient;

        axonsIndex++;
      }
      pathIndex--;
    }

    NeuronsActivation totalInputGradient =
        new NeuronsActivation(totalInputGradientMatrix, inputGradient.getFeatureOrientation());


    return new DirectedSynapsesGradientImpl(totalInputGradient, totalTrainableAxonsGradients,
        totalResidualGradient);
  }

  private AxonsGradient getTrainableAxonsGradient(AxonsActivation axonsActivation, 
      AxonsContext axonsContext, 
      NeuronsActivation gradientToBackPropagate) {
    Axons<?, ?, ?> axons = axonsActivation.getAxons();
    Matrix totalTrainableAxonsGradientMatrix = null;
    AxonsGradient totalTrainableAxonsGradient = null;
    
    if (axons instanceof TrainableAxons<?, ?, ?> && axons.isTrainable(axonsContext)) {

      LOGGER.debug("Calculating Axons Gradients");

      totalTrainableAxonsGradientMatrix = gradientToBackPropagate.getActivations()
          .mmul(axonsActivation.getPostDropoutInputWithPossibleBias().getActivationsWithBias());

      if (axonsContext.getRegularisationLambda() != 0) {

        LOGGER.debug("Calculating total regularisation Gradients");

        Matrix connectionWeightsCopy = axons.getDetachedConnectionWeights();

        Matrix firstRow = totalTrainableAxonsGradientMatrix.getRow(0);
        Matrix firstColumn = totalTrainableAxonsGradientMatrix.getColumn(0);

        totalTrainableAxonsGradientMatrix =
            totalTrainableAxonsGradientMatrix.addi(connectionWeightsCopy.muli(
                axonsContext.getRegularisationLambda()));

        if (axons.getLeftNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradientMatrix.putRow(0, firstRow);
        }
        if (axons.getRightNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradientMatrix.putColumn(0, firstColumn);
        }
        totalTrainableAxonsGradient = new AxonsGradientImpl((TrainableAxons<?, ?, ?>) axons, 
            totalTrainableAxonsGradientMatrix);
      }
    }
    return totalTrainableAxonsGradient == null ? null : 
      totalTrainableAxonsGradient;
  }
}
