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

package org.ml4j.nn;

import org.ml4j.nn.BackPropagation;
import org.ml4j.nn.DirectedNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Default implementation of ForwardPropagation.
 * 
 * @author Michael Lavelle
 */
public class ForwardPropagationImpl implements ForwardPropagation {
  
  private NeuronsActivation outputActivations;
  private List<DirectedLayerActivation> activations;
  
  /**
   * Create a new ForwardPropagation from the output activations at the
   * right hand side of a DirectedNeuralNetwork after a forward propagation.
   * 
   * @param activations All the DirectedLayerActivation instaces generated
   *        by the forward propagation.
   * @param outputActivations The output activations at the
   *        right hand side of a DirectedNeuralNetwork after a forward propagation.
   */
  public ForwardPropagationImpl(List<DirectedLayerActivation> activations, 
      NeuronsActivation outputActivations) {
    super();
    this.outputActivations = outputActivations;
    this.activations = activations;
  }

  @Override
  public NeuronsActivation getOutputs() {
    return outputActivations;
  }

  @Override
  public BackPropagation backPropagate(CostFunctionGradient neuronActivationGradients, 
      DirectedNeuralNetworkContext context) {
    
    List<DirectedLayerActivation> reversedActivations = new ArrayList<>();
    reversedActivations.addAll(activations);
  
    DirectedLayerGradient gradients = null;
    List<DirectedLayerGradient> gradientsRet = new ArrayList<>();
    Collections.reverse(reversedActivations);
    int layerIndex = reversedActivations.size() - 1;
    
    for (DirectedLayerActivation activation : reversedActivations) {

      DirectedLayerGradient gradient = null;
      if (gradients == null) {
        gradient = activation.backPropagate(neuronActivationGradients,
            context.getLayerContext(layerIndex));
      } else {
        gradient =
            activation.backPropagate(gradients, context.getLayerContext(layerIndex));
      }
      
      gradientsRet.add(gradient);
      gradients = gradient;
      layerIndex--;
    }
    
    Collections.reverse(gradientsRet);

    BackPropagation backPropagation =  new BackPropagationImpl(gradientsRet);
    if (context.getBackPropagationListener() != null) {
      context.getBackPropagationListener().onBackPropagation(backPropagation);
    }
    return backPropagation;
  }

  @Override
  public double getAverageRegularisationCost(DirectedNeuralNetworkContext context) {
    return getTotalRegularisationCost(context) / outputActivations.getActivations().getRows();
  }

  @Override
  public double getTotalRegularisationCost(DirectedNeuralNetworkContext context) {
    double totalRegularisationCost = 0d;
    int layerIndex = 0;
    for (DirectedLayerActivation activation : activations) {
      totalRegularisationCost = totalRegularisationCost + activation.getTotalRegularisationCost(
          context.getLayerContext(layerIndex++));
    }
    return totalRegularisationCost;
  }
  
  @Override
  public List<DirectedLayerActivation> getLayerActivations() {
    return activations;
  }
 
}
