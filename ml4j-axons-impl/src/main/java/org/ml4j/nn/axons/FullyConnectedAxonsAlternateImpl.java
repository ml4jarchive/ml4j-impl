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

package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of FullyConnectedAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class FullyConnectedAxonsAlternateImpl 
    extends TrainableAxonsBaseAlternateImpl<Neurons, Neurons, FullyConnectedAxons, AxonsConfig>
    implements FullyConnectedAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
		  FullyConnectedAxonsAlternateImpl.class);
  
  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public FullyConnectedAxonsAlternateImpl(Neurons leftNeurons, Neurons rightNeurons,
      MatrixFactory matrixFactory) {
    super(leftNeurons, rightNeurons, matrixFactory, new AxonsConfig());
  }
  
  
  public FullyConnectedAxonsAlternateImpl(Neurons leftNeurons, Neurons rightNeurons, 
        Matrix connectionWeights, Matrix leftToRightBiases) {
    super(leftNeurons, rightNeurons, connectionWeights, leftToRightBiases, new AxonsConfig());
  }
  
  public FullyConnectedAxonsAlternateImpl(Neurons leftNeurons, Neurons rightNeurons, 
		  Matrix connectionWeights, Matrix leftToRightBiases,  Matrix rightToLeftBiases) {
	    super(leftNeurons, rightNeurons, connectionWeights, leftToRightBiases, rightToLeftBiases, new AxonsConfig());
	  }
 

  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  protected Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory) {
   
    LOGGER.debug("Initialising FullyConnectedAxon weights...");
    EditableMatrix weights = (EditableMatrix)matrixFactory.createRandn(rightNeurons.getNeuronCountExcludingBias(), leftNeurons.getNeuronCountExcludingBias()).asInterrimMatrix();

    for (int i = 0; i < weights.getLength(); i++) {
    	if (Math.abs(weights.get(i) )> 1) {
    		// TODO
    		weights.put(i, ((float)((int)(weights.get(i) * 100))/100f));
    	} else {
    		weights.put(i, ((float)((int)(weights.get(i) * 100))/100f));
    	}
    }
    
    float scalingFactor = 
        (float)Math.sqrt(2d / ((float)leftNeurons.getNeuronCountIncludingBias()));
    
    weights.muli(scalingFactor);
    
    return weights;
  }
  
  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  protected InterrimMatrix createDefaultInitialLeftToRightBiases(MatrixFactory matrixFactory) {
   
    LOGGER.debug("Initialising FullyConnectedAxon left to right biases...");
    
    return matrixFactory.createZeros(rightNeurons.getNeuronCountExcludingBias(), 1).asInterrimMatrix();
    
  }
  
  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial connection weights
   */
  protected InterrimMatrix createDefaultInitialRightToLeftBiases(MatrixFactory matrixFactory) {
   
    LOGGER.debug("Initialising FullyConnectedAxon right to left biases...");
    
    return matrixFactory.createZeros(1, leftNeurons.getNeuronCountExcludingBias()).asInterrimMatrix();
    
  }

  @Override
  public FullyConnectedAxons dup() {
	  // TODO ML
    return new FullyConnectedAxonsAlternateImpl(leftNeurons, rightNeurons, 
    		axonWeights.getConnectionWeights().asInterrimMatrix(), axonWeights.getLeftToRightBiases().asInterrimMatrix());
  }
}
