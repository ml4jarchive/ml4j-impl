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

package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base Axons implementation.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the left hand side of these Axons
 * @param <R> The type of Neurons on the right hand side of these Axons
 * @param <A> The type of these Axons
 */
public abstract class TrainableAxonsBaseAlternateImpl<L extends Neurons, R extends Neurons, 
    A extends TrainableAxons<L, R, A>, C extends AxonsConfig> extends AxonsBaseAlternateImpl<L, R, A, C>
    implements TrainableAxons<L, R, A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(TrainableAxonsBaseAlternateImpl.class);

  protected AxonWeights axonWeights;
  //protected MatrixFactory matrixFactory1;
  
  @Override
  public boolean isTrainable(AxonsContext context) {
    return !context.isWithFreezeOut();
  }
  
  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param matrixFactory The matrix factory.
   * @param config The config for these Axons.
   */
  public TrainableAxonsBaseAlternateImpl(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory, C config) {
	  super(leftNeurons, rightNeurons, config);
	  //this.matrixFactory = matrixFactory;
	  if (leftNeurons.hasBiasUnit()) {
				  this.axonWeights = new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), rightNeurons.getNeuronCountExcludingBias(), createDefaultInitialConnectionWeights(matrixFactory), createDefaultInitialLeftToRightBiases(matrixFactory), null);
	  } else {
		  this.axonWeights = new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), rightNeurons.getNeuronCountExcludingBias(), createDefaultInitialConnectionWeights(matrixFactory), null, null);
	  }
  }

  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights.
   * @param config The config for these Axons.
   */
  public TrainableAxonsBaseAlternateImpl(L leftNeurons, R rightNeurons,
		  Matrix connectionWeights, Matrix leftToRightBiases, C config) {
	  super(leftNeurons, rightNeurons, config);
	  //this.matrixFactory = matrixFactory;
	  this.axonWeights = new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), rightNeurons.getNeuronCountExcludingBias(), connectionWeights, leftToRightBiases, null);
  }
  
  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights.
   * @param config The config for these Axons.
   */
  public TrainableAxonsBaseAlternateImpl(L leftNeurons, R rightNeurons,
      Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases, C config) {
	  super(leftNeurons, rightNeurons, config);
	  //this.matrixFactory = matrixFactory;
	  // TODO
	  this.axonWeights = new AxonWeightsImpl(leftNeurons.getNeuronCountExcludingBias(), rightNeurons.getNeuronCountExcludingBias(), connectionWeights, leftToRightBiases, rightToLeftBiases);
  }

  protected abstract Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory);
  protected abstract Matrix createDefaultInitialLeftToRightBiases(MatrixFactory matrixFactory);
  protected abstract Matrix createDefaultInitialRightToLeftBiases(MatrixFactory matrixFactory);
  
  @Override
  protected AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix inputDropoutMask, NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
	  	  
	  if (inputMatrix.getRows() != axonWeights.getOutputNeuronsCount()) {
	    	throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
	    			+ getRightNeurons().getNeuronCountIncludingBias() + " features including bias but references "
	    			+ inputMatrix.getRows() + " features including bias");
	    }
	  
	  		EditableMatrix outputMatrix = axonWeights.applyToGradient(inputMatrix.getActivations(axonsContext.getMatrixFactory())).asEditableMatrix();
	  
	 	    if (rightNeurons.hasBiasUnit()) {
	 	    	outputMatrix.addiColumnVector(axonWeights.getRightToLeftBiases());
	 	    }
	 	        
	 	    NeuronsActivation outputActivation =
	 	        new NeuronsActivationImpl(outputMatrix,
	 	        		featureOrientation);

	 	    return new AxonsActivationImpl(this, inputDropoutMask,
	 	    		inputMatrix,
	 	        outputActivation, leftNeurons, rightNeurons, true);
	   
  }


  @Override
  protected AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix inputDropoutMask, NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext) {
	  
	  
	  
	  if (inputMatrix.getFeatureCount() != axonWeights.getInputNeuronCount()) {
	    	throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
	    			+ getLeftNeurons().getNeuronCountExcludingBias() + " features excluding bias but references "
	    			+ inputMatrix.getFeatureCount() + " features excluding bias");
	    }
	  
	 
	  	EditableMatrix outputMatrix = axonWeights.applyToInput(inputMatrix.getActivations(axonsContext.getMatrixFactory())).asEditableMatrix();
	 
	    if (leftNeurons.hasBiasUnit()) {
	    	outputMatrix.addiColumnVector(axonWeights.getLeftToRightBiases());
	    }
	    
	    if (!isTrainable(axonsContext) && !inputMatrix.isImmutable()) {
			  inputMatrix.close();
		  }
	    	 

	    NeuronsActivation outputActivation =
	        new NeuronsActivationImpl(outputMatrix, 
	        		featureOrientation);

	    return new AxonsActivationImpl(this, inputDropoutMask,
	    		inputMatrix,
	        outputActivation, leftNeurons, rightNeurons, false);
  }


  @Override
  public Matrix getDetachedConnectionWeights() {
    LOGGER.debug("Duplicating connection weights");
    return axonWeights.getConnectionWeights().dup();
  }
  
  @Override
  public Matrix getDetachedLeftToRightBiases() {
    LOGGER.debug("Duplicating left to right biases");
    return axonWeights.getLeftToRightBiases().dup();
  }
  
  @Override
  public Matrix getDetachedRightToLeftBiases() {
    LOGGER.debug("Duplicating right to left biases");
    return axonWeights.getRightToLeftBiases().dup();
  }

  @Override
  public void adjustConnectionWeights(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
	  adjustConnectionWeights(adjustments, adjustmentDirection, false);
  }

  @Override
  public void adjustLeftToRightBiases(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
	  adjustLeftToRightBiases(adjustments, adjustmentDirection, false);
  }

  @Override
  public void adjustRightToLeftBiases(Matrix adjustments, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
	  adjustRightToLeftBiases(adjustments, adjustmentDirection, false);
  }

  protected void adjustConnectionWeights(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection,
		  boolean initialisation) {
	  

	  applyAdditionalConnectionWeightAdjustmentConstraints(adjustment);
	  axonWeights.adjustConnectionWeights(adjustment, adjustmentDirection, initialisation);
	  if (adjustment.getRows() != axonWeights.getOutputNeuronsCount()
			  || adjustment.getColumns() != axonWeights.getConnectionWeights().getColumns()) {
		  throw new IllegalArgumentException(
				  "Connection weights adjustment matrix is of dimensions: " + adjustment.getRows() + ","
						  + adjustment.getColumns() + " but connection weights matrix is of dimensions:"
						  + axonWeights.getConnectionWeights().getRows() + "," + axonWeights.getConnectionWeights().getColumns());
	  }

  }

  protected void adjustLeftToRightBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection,
		  boolean initialisation) {

	  if (adjustment.getRows() != axonWeights.getLeftToRightBiases().getRows()
			  || adjustment.getColumns() != axonWeights.getLeftToRightBiases().getColumns()) {
		  throw new IllegalArgumentException("Biases adjustment matrix is of dimensions: " + adjustment.getRows()
		  + "," + adjustment.getColumns() + " but biases matrix is of dimensions:"
		  + axonWeights.getLeftToRightBiases().getRows() + "," + axonWeights.getLeftToRightBiases().getColumns());
	  }

	  applyAdditionalLeftToRightBiasesAdjustmentConstraints(adjustment);
	  axonWeights.adjustLeftToRightBiases(adjustment, adjustmentDirection);
  }

  protected void adjustRightToLeftBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection,
		  boolean initialisation) {

	  if (adjustment.getRows() != axonWeights.getRightToLeftBiases().getRows()
			  || adjustment.getColumns() != axonWeights.getRightToLeftBiases().getColumns()) {
		  throw new IllegalArgumentException("Biases adjustment matrix is of dimensions: " + adjustment.getRows()
		  + "," + adjustment.getColumns() + " but biases matrix is of dimensions:"
		  + axonWeights.getRightToLeftBiases().getRows() + "," + axonWeights.getLeftToRightBiases().getColumns());
	  }

	  applyAdditionalRightToLeftBiasesAdjustmentConstraints(adjustment);
	  axonWeights.adjustRightToLeftBiases(adjustment, adjustmentDirection);
  }
  
  @Override
  protected boolean isLeftInputDropoutSupported() {
	return true;
  }

  protected void applyAdditionalConnectionWeightAdjustmentConstraints(Matrix adjustment) {
	  // No-op by default
  }

  protected void applyAdditionalLeftToRightBiasesAdjustmentConstraints(Matrix adjustment) {
	  // No-op by default
  }

  protected void applyAdditionalRightToLeftBiasesAdjustmentConstraints(Matrix adjustment) {
	  // No-op by default
  }
}
