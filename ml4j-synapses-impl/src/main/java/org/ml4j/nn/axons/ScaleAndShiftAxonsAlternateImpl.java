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
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive sparse-matrix implementation of ScaleAndShiftAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ScaleAndShiftAxonsAlternateImpl 
    extends TrainableAxonsBase<Neurons, Neurons, ScaleAndShiftAxons, ScaleAndShiftAxonsConfig>
    implements ScaleAndShiftAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(
      ScaleAndShiftAxonsAlternateImpl.class);
  
  /**
   * @param leftNeurons The neurons whose activations we want to scale and shift.
   * @param rightNeurons The target neurons.
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   * @param config The config for these Axons.
   */
  public ScaleAndShiftAxonsAlternateImpl(Neurons leftNeurons, Neurons rightNeurons,
      MatrixFactory matrixFactory, ScaleAndShiftAxonsConfig config) {
    super(new Neurons(1, true), rightNeurons, matrixFactory, config);
    if (!leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
    }
    if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Left neurons and right neurons are not compatible sizes:"
          + leftNeurons.getNeuronCountExcludingBias() + ":"
          + rightNeurons.getNeuronCountExcludingBias());
    }
    if (!leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
    }
  }
  
  protected ScaleAndShiftAxonsAlternateImpl(Neurons leftNeurons, Neurons rightNeurons, 
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
  }
 
  @Override
  protected ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory) {
    return null;
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
    
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException("Right neurons should not contain bias unit");
    }
    
    Matrix weights = matrixFactory.createZeros(2,
        rightNeurons.getNeuronCountIncludingBias());
    
    weights.putRow(0, config.getScaleRowVector());
    weights.putRow(1, config.getShiftRowVector());
  
    return weights;
  }
  
  

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
   
    NeuronsActivationWithPossibleBiasUnit inputActivation = 
        new NeuronsActivationWithPossibleBiasUnit(leftNeuronsActivation.getActivations(),
            false, leftNeuronsActivation.getFeatureOrientation(), false);
    
    Matrix scaleRowVector = getScaleRowVector();
    Matrix shiftRowVector = getShiftRowVector();

    Matrix scaleMatrix = axonsContext.getMatrixFactory().createMatrix(
        inputActivation.getActivationsWithoutBias().getRows(), scaleRowVector.getColumns());
    Matrix shiftMatrix = axonsContext.getMatrixFactory().createMatrix(
        inputActivation.getActivationsWithoutBias().getRows(), shiftRowVector.getColumns());
    for (int i = 0; i < scaleMatrix.getRows(); i++) {
      scaleMatrix.putRow(i, scaleRowVector);
    }
    for (int i = 0; i < shiftMatrix.getRows(); i++) {
      shiftMatrix.putRow(i, shiftRowVector);
    }
    Matrix result = inputActivation.getActivationsWithoutBias().mul(scaleMatrix).add(shiftMatrix);
        
    NeuronsActivation output = new NeuronsActivation(result, 
        inputActivation.getFeatureOrientation());
     
    
    return new AxonsActivationImpl(this, null, inputActivation, output);
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
    
    // Matrix xhat = previousLeftToRightActivation.getOutput().getActivations();
    Matrix dout = rightNeuronsActivation.getActivations().transpose();

    //Matrix dgamma = xhat.mul(dout).transpose().rowSums().transpose();

    // Matrix dbeta = dout.transpose().rowSums().transpose();

    Matrix scaleMatrix 
        =  axonsContext.getMatrixFactory().createMatrix(dout.getRows(), dout.getColumns());
    for (int r = 0; r < scaleMatrix.getRows(); r++) {
      scaleMatrix.putRow(r, getScaleRowVector());
      
    }
     
    
    Matrix nonBiasInputs = dout.mul(scaleMatrix);
    //Matrix biasInputs = dout.rowSums();
    //Matrix result = biasInputs.appendHorizontally(nonBiasInputs);
    
    NeuronsActivation output = 
        new NeuronsActivation(nonBiasInputs.transpose(), 
        rightNeuronsActivation.getFeatureOrientation());
   
    
    NeuronsActivationWithPossibleBiasUnit inputActivation = 
        new NeuronsActivationWithPossibleBiasUnit(rightNeuronsActivation.getActivations(),
            false, rightNeuronsActivation.getFeatureOrientation(), false);
   
    Matrix inputDropoutMask = null;
    return new AxonsActivationImpl(this, inputDropoutMask, 
        inputActivation.withBiasUnit(true, 
            axonsContext),
        output);

  }

  @Override
  public ScaleAndShiftAxons dup() {
    return new ScaleAndShiftAxonsAlternateImpl(leftNeurons, rightNeurons, 
        connectionWeights.dup(), connectionWeightsMask, config);
  }

  @Override
  public Matrix getScaleRowVector() {
    return connectionWeights.getRow(0);
  }

  @Override
  public Matrix getShiftRowVector() {
    return connectionWeights.getRow(1);
  }
}
