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

package org.ml4j.nn.activationfunctions;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive prototype implementation of a Batch Norm Activation Function - calculates mean and 
 * variance for each input feature and normalises the input.
 * 
 * @author Michael Lavelle
 *
 */
public class BatchNormActivationFunction implements DifferentiableActivationFunction {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(BatchNormActivationFunction.class);

  @Override
  public NeuronsActivation activate(NeuronsActivation input, NeuronsActivationContext context) {
    LOGGER.debug("Activating through BatchNormActivationFunction");
    
    NeuronsActivation inputWithoutBias = input.withBiasUnit(false, context);
    
    Matrix meanMatrix = getMeanMatrix(input, context.getMatrixFactory());
    
    Matrix normedInput = input.getActivations().sub(meanMatrix);
   
    normedInput = divi(normedInput, getVarianceMatrix(input, context.getMatrixFactory(), 
        meanMatrix.getRow(0)));
    
    return inputWithoutBias.withBiasUnit(input.isBiasUnitIncluded(), context);
  }
  
  /**
   * Temporary method for prototype purposes until divi is implemented in Matrix interface.
   * 
   * @param matrix The input matrix.
   * @param by The matrix we are dividing the input matrix by.
   * @return The input matrix with amended inline entries for the division result.
   */
  private Matrix divi(Matrix matrix, Matrix by) {
    for (int r = 0; r < matrix.getRows(); r++) {
      for (int c = 0; c < matrix.getColumns(); c++) {
        matrix.put(r, c, matrix.get(r, c) / by.get(r, c));
      }
    }
    return matrix;
  }

  /**
   * Naive implementation to construct a variance row vector with an entry for each feature.
   * 
   * @param matrix The input matrix
   * @param matrixFactory The matrix factory.
   * @param meanRowVector The mean row vector.
   * @return A row vector the the variances.
   */
  private Matrix getVarianceRowVector(Matrix matrix, MatrixFactory matrixFactory, 
      Matrix meanRowVector) {
    Matrix rowVector = matrixFactory.createMatrix(1, matrix.getColumns());
    for (int c = 0; c < matrix.getColumns(); c++) {
      double total = 0d;
      double count = 0;
      for (int r = 0; r < matrix.getRows(); r++) {
        double diff = (matrix.get(r, c) - meanRowVector.get(c));
        total = total + diff * diff;
        count++;
      }
      double variance = total / (count - 1);

      double epsilion = 0.00000001;
      double varianceVal = Math.sqrt(variance * variance + epsilion);
      rowVector.put(0, c, varianceVal) ;
    }
    return rowVector;
  }

  private Matrix getMeanRowVector(Matrix matrix, MatrixFactory matrixFactory) {
    Matrix rowVector = matrixFactory.createMatrix(1, matrix.getColumns());
    for (int c = 0; c < matrix.getColumns(); c++) {
      double mean = matrix.getColumn(c).sum() / matrix.getRows();
      rowVector.put(0, c, mean);
    }
    return rowVector;
  }
  
  private Matrix getMeanMatrix(NeuronsActivation input, MatrixFactory matrixFactory) {
    
    if (input.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException("Only input without bias supported");
    }
    
    Matrix meanMatrix = matrixFactory.createMatrix(
        input.getActivations().getRows(), input.getActivations().getColumns());
    for (int r = 0; r < meanMatrix.getRows(); r++) {
      meanMatrix.putRow(r, getMeanRowVector(input.getActivations(), matrixFactory));
    }
    
    return meanMatrix;
  }
  
  private Matrix getVarianceMatrix(NeuronsActivation input, 
      MatrixFactory matrixFactory, Matrix meanRowVector) {

    if (input.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException("Only input without bias supported");
    }
    
    Matrix varianceMatrix = matrixFactory.createMatrix(input.getActivations().getRows(),
        input.getActivations().getColumns());
    for (int r = 0; r < varianceMatrix.getRows(); r++) {
      varianceMatrix.putRow(r,
          getVarianceRowVector(input.getActivations(), matrixFactory, meanRowVector));
    }

    return varianceMatrix;
  }

  @Override
  public NeuronsActivation activationGradient(NeuronsActivation outputActivation,
      NeuronsActivationContext context) {
    
    if (outputActivation.isBiasUnitIncluded()) {
      throw new IllegalArgumentException(
          "Activation gradient of activations with bias unit not supported");
    }
    
    LOGGER.debug("Performing batch norm gradient of NeuronsActivation");
   
    Matrix meanRowVector = getMeanRowVector(outputActivation.getActivations(), 
        context.getMatrixFactory());
    
    Matrix gradient = divi(context.getMatrixFactory()
        .createOnes(outputActivation.getActivations().getRows(), 
        outputActivation.getActivations().getColumns()), 
        getVarianceMatrix(outputActivation, context.getMatrixFactory(), meanRowVector));
   
    return new NeuronsActivation(
        gradient,
        outputActivation.isBiasUnitIncluded(), 
        outputActivation.getFeatureOrientation());
  }
}
