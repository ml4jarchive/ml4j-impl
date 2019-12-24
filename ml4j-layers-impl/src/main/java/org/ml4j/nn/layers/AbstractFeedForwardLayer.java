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

package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A default base implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of primary Axons in this FeedForwardLayer.
 */
public abstract class AbstractFeedForwardLayer<A extends Axons<?, ?, ?>, 
    L extends FeedForwardLayer<A, L>> 
    implements FeedForwardLayer<A, L> {

/**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(AbstractFeedForwardLayer.class);

  protected MatrixFactory matrixFactory;
  
  protected DirectedComponentChain<NeuronsActivation, ? extends DefaultChainableDirectedComponent<?, ?>, ?, ?> componentChain;
  protected TrailingActivationFunctionDirectedComponentChain 
  		trailingActivationFunctionDirectedComponentChain;
  
 
  /**
   * @param primaryAxons The primary Axons
   * @param activationFunction The primary activation function
   * @param matrixFactory The matrix factory
   * @param withBatchNorm Whether to enable batch norm.
   */
  protected AbstractFeedForwardLayer(DirectedComponentFactory directedComponentFactory, DirectedComponentChain<NeuronsActivation, ? extends DefaultChainableDirectedComponent<?,?>, ?, ?> componentChain, MatrixFactory matrixFactory) {
	  this.componentChain = componentChain;
	  List<DefaultChainableDirectedComponent<?, ?>> chainableComponents = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> component : componentChain.getComponents()) {
			chainableComponents.addAll(component.decompose());
		}
	  this.trailingActivationFunctionDirectedComponentChain = new TrailingActivationFunctionDirectedComponentChainImpl(directedComponentFactory, chainableComponents);
	  this.matrixFactory = matrixFactory;
  }
  
  
  @Override
  public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug(directedLayerContext.toString() + ":Forward propagating through layer");
    
	DirectedComponentsContext componentsContext = new DirectedComponentsContextImpl(directedLayerContext.getMatrixFactory(), directedLayerContext.isTrainingContext()); 

    TrailingActivationFunctionDirectedComponentChainActivation activation = trailingActivationFunctionDirectedComponentChain.forwardPropagate(inputNeuronsActivation, componentsContext);
    
	return new DirectedLayerActivationImpl(this, activation, directedLayerContext);   
  }
  
  @Override
  public DirectedLayerContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
	  return directedComponentsContext.getContext(this, () -> new DirectedLayerContextImpl(componentIndex, matrixFactory, directedComponentsContext.isTrainingContext()));
  }
  

  @Override
  public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
	System.out.println("blah");
	  for (DefaultChainableDirectedComponent<?, ?> b : getComponents()) {
		System.out.println(b);
	}
	return getComponents().stream().flatMap(c -> c.decompose().stream()).collect((Collectors.toList()));
  }

 
  @Override
  public DirectedComponentType getComponentType() {
	  return DirectedComponentType.LAYER;
  }

  
}
