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

import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.layers.DirectedLayer;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.DirectedLayerGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Default implementation of DirectedLayerActivation.
 * 
 * 
 * @author Michael Lavelle
 */
public class DirectedLayerActivationImpl implements DirectedLayerActivation {


  private static final Logger LOGGER = LoggerFactory.getLogger(DirectedLayerActivationImpl.class);

  private NeuronsActivation outputActivation;
  private List<DirectedSynapsesActivation> synapseActivations;
  private DirectedLayer<?, ?> layer;

  /**
   * @param layer The layer.
   * @param synapseActivations The activations
   * @param outputActivation The output
   */
  public DirectedLayerActivationImpl(DirectedLayer<?, ?> layer,
      List<DirectedSynapsesActivation> synapseActivations, NeuronsActivation outputActivation) {
    this.outputActivation = outputActivation;
    this.synapseActivations = synapseActivations;
    this.layer = layer;
  }

  @Override
  public NeuronsActivation getOutput() {
    return outputActivation;
  }

  @Override
  public DirectedLayerGradient backPropagate(DirectedLayerGradient activationGradient,
      DirectedLayerContext layerContext) {

    LOGGER.debug(layerContext.toString() + ":"
        + "Back propagating outer layer gradient through layer activation....");

    List<DirectedSynapsesActivation> reversedSynapseActivations =
        new ArrayList<DirectedSynapsesActivation>();
    reversedSynapseActivations.addAll(synapseActivations);
    Collections.reverse(reversedSynapseActivations);

    List<DirectedSynapsesGradient> outerSynapsesGradients =
        activationGradient.getSynapsesGradients();

    DirectedSynapsesGradient outerSynapsesGradient =
        outerSynapsesGradients.get(outerSynapsesGradients.size() - 1);

    List<DirectedSynapsesGradient> previousGradientList = new ArrayList<>();

    return backPropagateAndAddToSynapseGradientList(previousGradientList, outerSynapsesGradient,
        reversedSynapseActivations, layerContext);
  }

  @Override
  public DirectedLayerGradient backPropagate(CostFunctionGradient activationGradient,
      DirectedLayerContext layerContext) {

    LOGGER.debug(layerContext.toString() + ":"
        + "Back propagating cost function gradient through layer activation....");

    List<DirectedSynapsesActivation> reversedSynapseActivations =
        new ArrayList<DirectedSynapsesActivation>();
    reversedSynapseActivations.addAll(synapseActivations);
    Collections.reverse(reversedSynapseActivations);
    int index = reversedSynapseActivations.size() - 1;
    DirectedSynapsesActivation outerActivation = reversedSynapseActivations.get(0);

    DirectedSynapsesContext context = layerContext.getSynapsesContext(index);

    DirectedSynapsesGradient grad =
        outerActivation.backPropagate(activationGradient, context);
    List<DirectedSynapsesGradient> acts = new ArrayList<>();
    acts.add(grad);
    List<DirectedSynapsesActivation> remainingActivations = new ArrayList<>();
    for (int i = 1; i < reversedSynapseActivations.size(); i++) {
      remainingActivations.add(reversedSynapseActivations.get(i));
    }
    return backPropagateAndAddToSynapseGradientList(acts, grad, remainingActivations, layerContext);
  }

  protected DirectedLayerGradient backPropagateAndAddToSynapseGradientList(
      List<DirectedSynapsesGradient> synapseGradientList,
      DirectedSynapsesGradient outerSynapsesGradient,
      List<DirectedSynapsesActivation> activationsToBackPropagateThrough,
      DirectedLayerContext layerContext) {

    int index = activationsToBackPropagateThrough.size() - 1;
    NeuronsActivation finalGrad = null;
    DirectedSynapsesGradient synapsesGradient = outerSynapsesGradient;
    for (DirectedSynapsesActivation synapsesActivation : activationsToBackPropagateThrough) {
      
      DirectedSynapsesContext context = layerContext.getSynapsesContext(index);

      synapsesGradient =
          synapsesActivation.backPropagate(synapsesGradient, context);

      synapseGradientList.add(synapsesGradient);
      finalGrad = synapsesGradient.getOutput();
      index--;
    }

    return new DirectedLayerGradientImpl(finalGrad, synapseGradientList);
  }


  @Override
  public DirectedLayer<?, ?> getLayer() {
    return layer;
  }

  @Override
  public double getAverageRegularistationCost(DirectedLayerContext layerContext) {
    return getTotalRegularisationCost(layerContext)
        / outputActivation.getActivations().getRows();
  }

  @Override
  public double getTotalRegularisationCost(DirectedLayerContext layerContext) {
    double totalRegularisationCost = 0d;
    int synapsesIndex = 0;
    for (DirectedSynapsesActivation activation : synapseActivations) {
      DirectedSynapsesContext synapsesContext = layerContext.getSynapsesContext(synapsesIndex);
      totalRegularisationCost =
          totalRegularisationCost + activation.getTotalRegularisationCost(synapsesContext);
      synapsesIndex++;
    }
    return totalRegularisationCost;
  }

  @Override
  public List<DirectedSynapsesActivation> getSynapsesActivations() {
    return synapseActivations;
  }
}
