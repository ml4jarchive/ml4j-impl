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

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.defaults.DefaultChainableDirectedComponentActivationAdapter;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of ForwardPropagation.
 * 
 * @author Michael Lavelle
 */
public class ForwardPropagationImpl implements ForwardPropagation {
  
  private TrailingActivationFunctionDirectedComponentChainActivation activationChain;
  
  /**
   * Create a new ForwardPropagation from the output activations at the
   * right hand side of a DirectedNeuralNetwork after a forward propagation.
   * 
   * @param activations All the DirectedLayerActivation instaces generated
   *        by the forward propagation.
   * @param outputActivations The output activations at the
   *        right hand side of a DirectedNeuralNetwork after a forward propagation.
   */
  public ForwardPropagationImpl(TrailingActivationFunctionDirectedComponentChainActivation activationChain) {
    super();
    this.activationChain = activationChain;
  }

  @Override
  public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> arg0) {
	  return activationChain.backPropagate(arg0);
  }

  @Override
  public BackPropagation backPropagate(CostFunctionGradient neuronActivationGradients, 
      DirectedNeuralNetworkContext context) {
    BackPropagation backPropagation =  new BackPropagationImpl(activationChain.backPropagate(neuronActivationGradients));
    if (context.getBackPropagationListener() != null) {
      context.getBackPropagationListener().onBackPropagation(backPropagation);
    }
    return backPropagation;
  }

  @Override
  public float getTotalRegularisationCost(DirectedNeuralNetworkContext context) {
   float totalRegularisationCost = 0;
 
    for (ChainableDirectedComponentActivation<NeuronsActivation> activation : activationChain.getActivations()) {
    	for (ChainableDirectedComponentActivation<NeuronsActivation> decomposedActivation : activation.decompose()) {
	    	if (decomposedActivation instanceof DirectedAxonsComponentActivation) {
	    		DirectedAxonsComponentActivation axonsActivation = (DirectedAxonsComponentActivation)decomposedActivation;
	    		 totalRegularisationCost = totalRegularisationCost + axonsActivation.getTotalRegularisationCost();
	    	}
    	}
    }
  
    return totalRegularisationCost;
  }

@Override
public List<DefaultChainableDirectedComponentActivation> decompose() {
	return activationChain.decompose().stream().map(c -> new DefaultChainableDirectedComponentActivationAdapter(c)).collect(Collectors.toList());
}

@Override
public NeuronsActivation getOutput() {
    return activationChain.getOutput();
}

}
