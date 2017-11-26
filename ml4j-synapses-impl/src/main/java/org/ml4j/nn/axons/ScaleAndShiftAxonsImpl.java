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

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive sparse-matrix implementation of ScaleAndShiftAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ScaleAndShiftAxonsImpl 
    extends TrainableAxonsBase<Neurons, Neurons, ScaleAndShiftAxons, ScaleAndShiftAxonsConfig>
    implements ScaleAndShiftAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      ScaleAndShiftAxonsImpl.class);
    
  /**
   * @param leftNeurons The left hand neurons whose activations we want to scale and shift.
   * @param rightNeurons The right hand target neurons..
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   * @param config The config for these Axons.
   */
  public ScaleAndShiftAxonsImpl(Neurons leftNeurons, Neurons rightNeurons,
      MatrixFactory matrixFactory, ScaleAndShiftAxonsConfig config) {
    super(new Neurons(leftNeurons.getNeuronCountExcludingBias(),true), 
        rightNeurons, matrixFactory, config);
    if (!leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
    }
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException("Right neurons should not contain bias unit");
    }
    if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Left neurons and right neurons are not compatible sizes");
    }
  }
  
  
  protected ScaleAndShiftAxonsImpl(Neurons leftNeurons, Neurons rightNeurons, 
        Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask, 
        ScaleAndShiftAxonsConfig config) {
    super(leftNeurons, rightNeurons, connectionWeights, connectionWeightsMask, config);
    if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Left neurons and right neurons are not compatible sizes");
    }
    if (!leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
    }
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException("Right neurons should not contain bias unit");
    }
  }
 
  @Override
  protected ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory) {
    return new ScaleAndShiftWeightsMask(matrixFactory, leftNeurons, rightNeurons);
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
    
    Matrix weights = matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountExcludingBias());
    
    if (getLeftNeurons().hasBiasUnit()) {
      weights.putRow(0, config.getShiftRowVector());
    }
  
    for (int i = 0; i < config.getScaleRowVector().getColumns(); i++) {
      weights.put(i + 1, i, config.getScaleRowVector().get(0, i));
    }
 
    return weights;
  }

  @Override
  public ScaleAndShiftAxons dup() {
    return new ScaleAndShiftAxonsImpl(leftNeurons, rightNeurons, 
        connectionWeights.dup(), connectionWeightsMask, config);
  }

  @Override
  public Matrix getScaleRowVector() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix getShiftRowVector() {
    return connectionWeights.getRow(0);
  }
}
