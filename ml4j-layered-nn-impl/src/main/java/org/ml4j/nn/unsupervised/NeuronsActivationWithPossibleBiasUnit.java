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

package org.ml4j.nn.unsupervised;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Encapsulates the activation activities of a set of Neurons.
 * 
 * @author Michael Lavelle
 */
public class NeuronsActivationWithPossibleBiasUnit {

  private static final Logger LOGGER = 
        LoggerFactory.getLogger(NeuronsActivationWithPossibleBiasUnit.class);
  
  /**
   * The matrix of activations.
   */
  private Matrix activations;
  
  /**
   * Defines whether the features of the activations are represented by the columns
   * or the rows of the activations Matrix.
   */
  private NeuronsActivationFeatureOrientation featureOrientation;

  
  /**
   * Whether the activations include an activation from a bias unit.
   */
  private boolean biasUnitIncluded;
  

  /**
   * Constructs a NeuronsActivation instance from a matrix of activations.
   * 
   * @param activations A matrix of activations
   * @param biasUnitIncluded Whether a bias unit is included in the activation features
   * @param featureOrientation The orientation of the features of the activation matrix
   * @param resetBiasValues Whether to reset the bias values of the activations matrix
   */
  public NeuronsActivationWithPossibleBiasUnit(Matrix activations, boolean biasUnitIncluded,
      NeuronsActivationFeatureOrientation featureOrientation, boolean resetBiasValues) {
    LOGGER.debug("Creating new NeuronsActivationWithPossibleBiasUnit");
    this.activations = activations;
    this.biasUnitIncluded = biasUnitIncluded;
    this.featureOrientation = featureOrientation;
    if (biasUnitIncluded) {
      if (resetBiasValues) {
        resetBiasActivations(activations, featureOrientation);
      }
      validateBiasActivations(activations, featureOrientation);
    }
  }

  /**
   * Obtain the feature orientation of the Matrix representing the activations - whether the
   * features are represented by the rows or the columns.
   * 
   * @return the feature orientation of the Matrix representing the activations - whether the
   *         features are represented by the rows or the columns
   */
  public NeuronsActivationFeatureOrientation getFeatureOrientation() {
    return featureOrientation;
  }


  public Matrix getActivations() {
    return activations;
  }


  /**
   * Getting activations without bias.
   * 
   * @return activations without bias.
   */
  public Matrix getActivationsWithoutBias() {

    if (biasUnitIncluded) {
      if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
        return removeFirstColumn(this.activations);
      } else {
        return removeFirstRow(this.activations);
      }
    } else {
      return activations;
    }
  }
  
  public Matrix getActivationsWithBias() {
    return activations;
  }

  /**
   * Indicates whether the features represented by this NeuronsActivation include a bias unit.
   * 
   * @return Whether the features represented by this NeuronsActivation include a bias unit
   */
  public boolean isBiasUnitIncluded() {
    return biasUnitIncluded;
  }

  /**
   * Obtain the number of features ( including any bias ) represented by this NeuronsActivation.
   * 
   * @return the number of features ( including any bias ) represented by this NeuronsActivation.
   */
  public int getFeatureCountIncludingBias() {

    if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      int featureCount = activations.getColumns();
      return featureCount;
    } else {
      int featureCount = activations.getRows();
      return featureCount;
    }
  }

  /**
   * Obtain the number of features ( excluding any bias ) represented by this NeuronsActivation.
   *
   * @return the number of features ( excluding any bias ) represented by this NeuronsActivation.
   */
  public int getFeatureCountExcludingBias() {
    int featureCountIncludingBias = getFeatureCountIncludingBias();
    return biasUnitIncluded ? (featureCountIncludingBias - 1) : featureCountIncludingBias;
  }

  /**
   * Returns this NeuronsActivation ensuring the presence of a bias unit is consistent with the 
   * requested withBiasUnit parameter.
   * 
   * @param withBiasUnit Whether a bias unit should be included in the returned activation
   * @param neuronsActivationContext The activation context
   * @return this NeuronsActivation ensuring the presence of a bias unit is consistent with the 
   *         requested withBiasUnit parameter.
   */
  public NeuronsActivationWithPossibleBiasUnit withBiasUnit(boolean withBiasUnit,
      NeuronsActivationContext neuronsActivationContext) {

    MatrixFactory matrixFactory = neuronsActivationContext.getMatrixFactory();

    if (isBiasUnitIncluded()) {
      if (withBiasUnit) {
        return this;
      } else {

        if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {

          LOGGER.debug("Removing bias unit from activations");
          
          return new NeuronsActivationWithPossibleBiasUnit(removeFirstColumn(activations), false,
              NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, false);

        } else if (featureOrientation 
            == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
          LOGGER.debug("Removing bias unit from activations");
        
          return new NeuronsActivationWithPossibleBiasUnit(removeFirstRow(activations), false,
              NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
        } else {
          throw new IllegalStateException(
              "Unsupported feature orientation type:" + featureOrientation);
        }
      
      }
    } else {
      if (withBiasUnit) {

        if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {

          Matrix bias = matrixFactory.createOnes(activations.getRows(), 1);
          LOGGER.debug("Adding bias unit to activations");
          Matrix activationsWithBias = bias.appendHorizontally(activations);
          return new NeuronsActivationWithPossibleBiasUnit(activationsWithBias, true,
              NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET, false);

        } else if (featureOrientation 
            == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
          Matrix bias = matrixFactory.createOnes(1, activations.getColumns());
          LOGGER.debug("Adding bias unit to activations");
          Matrix activationsWithBias = bias.appendVertically(activations);
          return new NeuronsActivationWithPossibleBiasUnit(activationsWithBias, true,
              NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
        } else {
          throw new IllegalStateException(
              "Unsupported feature orientation type:" + featureOrientation);
        }
      } else {
        return this;
      }
    }
  }
  
  
  private void validateBiasActivations(Matrix activations,
      NeuronsActivationFeatureOrientation featureOrientation) {
    if (!biasUnitIncluded) {
      throw new IllegalStateException("Cannot validate bias activations as bias unit not included");
    }
    LOGGER.debug("Validating bias activations");
    if (NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET == featureOrientation) {
      Matrix firstColumn = activations.getColumn(0);
      for (int r = 0; r < activations.getRows(); r++) {
        if (firstColumn.get(r, 0) != 1d) {
          throw new IllegalArgumentException("Values of bias unit is not 1");
        }
      }
    } else if (NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET == featureOrientation) {
      Matrix firstRow = activations.getRow(0);
      for (int c = 0; c < activations.getColumns(); c++) {
        if (firstRow.get(0, c) != 1d) {
          throw new IllegalArgumentException("Values of bias unit is not 1");
        }
      }
    } else {
      throw new IllegalStateException(
          "Unsupported feature orientation type:" + featureOrientation);
    }
  }
  
  private Matrix removeFirstColumn(Matrix activations) {
    int[] rows = new int[activations.getRows()];
    for (int r = 0; r < rows.length; r++) {
      rows[r] = r;
    }
    int[] cols = new int[activations.getColumns() - 1];
    for (int c = 0; c < cols.length; c++) {
      cols[c] = c + 1;
    }
    return activations.get(rows, cols);
  }
  
  private Matrix removeFirstRow(Matrix activations) {
    int[] rows = new int[activations.getRows() - 1];
    for (int r = 0; r < rows.length; r++) {
      rows[r] = r + 1;
    }
    int[] cols = new int[activations.getColumns()];
    for (int c = 0; c < cols.length; c++) {
      cols[c] = c;
    }
    return activations.get(rows, cols);
  }
  
  private void resetBiasActivations(Matrix activations,
      NeuronsActivationFeatureOrientation featureOrientation) {
    if (!biasUnitIncluded) {
      throw new IllegalStateException("Cannot reset bias activations as bias unit not included");
    }
    LOGGER.debug("Resetting bias activations");
    if (NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET == featureOrientation) {
      for (int r = 0; r < activations.getRows(); r++) {
        activations.asEditableMatrix().put(r, 0, 1);
      }
    }
    if (NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET == featureOrientation) {
      for (int c = 0; c < activations.getColumns(); c++) {
        activations.asEditableMatrix().put(0, c, 1);
      }
    }
  }
}
