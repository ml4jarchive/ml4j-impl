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
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
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
public abstract class AxonsBaseAlternateImpl<L extends Neurons, R extends Neurons, 
    A extends Axons<L, R, A>, C extends AxonsConfig>
    implements Axons<L, R, A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(AxonsBaseAlternateImpl.class);

  protected L leftNeurons;
  protected R rightNeurons;
  protected C config;
  
  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param matrixFactory The matrix factory.
   * @param config The config for these Axons.
   */
  public AxonsBaseAlternateImpl(L leftNeurons, R rightNeurons, C config) {
    this.config = config;
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
  }


  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }
  
  protected abstract boolean isLeftInputDropoutSupported();
 
 
  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
	  
	  
	  if (!(leftNeuronsActivation instanceof ImageNeuronsActivation)) {
		  //throw new IllegalArgumentException();
	  }
    	  
    LOGGER.debug("Pushing left to right through Axons:" 
    + leftNeuronsActivation.getFeatureCount()+ ":" + leftNeuronsActivation.getExampleCount());
    if (leftNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    Matrix outputDropoutMask = null;
    Matrix inputDropoutMask = createLeftInputDropoutMask(leftNeuronsActivation, axonsContext);

    Matrix previousInputDropoutMask = previousRightToLeftActivation == null ? null
        : previousRightToLeftActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null) {
      LOGGER.debug("Transposing previous right to left input dropout mask");
      outputDropoutMask = previousInputDropoutMask.transpose();
    }
        
    // TODO MON
    Matrix inputMatrix = null;

    if (inputDropoutMask != null) {
    	float postDropoutScaling = getLeftInputPostDropoutScaling(axonsContext);

    	if (postDropoutScaling != 1) {
    		LOGGER.debug("Applying input dropout mask");

    		LOGGER.debug("Scaling post dropout left to right non-bias input");

    		inputMatrix = leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory())
    				.mul(inputDropoutMask).asEditableMatrix().muli(postDropoutScaling);

    	} else {
    		inputMatrix = leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).mul(inputDropoutMask);
    	}
    } else {
    	//inputMatrix = leftNeuronsActivation.getActivations();
    }
    AxonsActivation axonsActivation = doPushLeftToRight(leftNeuronsActivation, inputDropoutMask, leftNeuronsActivation.getFeatureOrientation(), axonsContext);
      
    
    if (outputDropoutMask != null) {
      LOGGER.debug("Applying left to right output dropout mask");
      axonsActivation.applyOutputDropoutMask(outputDropoutMask);
      //outputDropoutMask.close();
    }
    
    //LOGGER.debug("End Pushing left to right through Axons:" 
    //	    + axonsActivation.getOutput().getActivations().getRows() + ":" + axonsActivation.getOutput().getActivations().getColumns() + ":" +  leftNeuronsActivation.getActivations().getColumns() + leftNeuronsActivation.getActivations().getClass().getName());
       
    return axonsActivation;
  }
  
  protected abstract AxonsActivation doPushLeftToRight(NeuronsActivation inputMatrix, Matrix previousRightToLeftInputDropoutMask, NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext);
  protected abstract AxonsActivation doPushRightToLeft(NeuronsActivation inputMatrix, Matrix previousLeftToRightInputDropoutMask, NeuronsActivationFeatureOrientation featureOrientation, AxonsContext axonsContext);
  
  /**
   * Return the dropout mask for left hand side input.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the left hand side of these Axons
   */
  protected Matrix createLeftInputDropoutMask(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {

    double leftHandInputDropoutKeepProbability =
        axonsContext.getLeftHandInputDropoutKeepProbability();
    if (leftHandInputDropoutKeepProbability == 1) {
      return null;
    } else {
    	
      if (!isLeftInputDropoutSupported()) {
    	  throw new IllegalStateException("Left input dropout is not supported for these axons");
      }

      LOGGER.debug("Creating left input dropout mask");

      EditableMatrix dropoutMask = axonsContext.getMatrixFactory().createZeros(
          leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).getRows(),
          leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).getColumns()).asEditableMatrix();
      for (int i = 0; i < dropoutMask.getRows(); i++) {
        for (int j = 0; j < dropoutMask.getColumns(); j++) {
          if (Math.random() < leftHandInputDropoutKeepProbability) {
            dropoutMask.put(i, j, 1);
          }
        }
      }
      return dropoutMask;

    }
  }

  /**
   * Return the scaling required due to left-hand side input dropout.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected float getLeftInputPostDropoutScaling(AxonsContext axonsContext) {
	  float dropoutKeepProbability = axonsContext.getLeftHandInputDropoutKeepProbability();
    if (dropoutKeepProbability == 0) {
      throw new IllegalArgumentException("Dropout keep probability cannot be set to 0");
    }
    return 1f / dropoutKeepProbability;
  }

  /**
   * Return the scaling required due to right-hand side input dropout. This is not yet supported, so
   * we return 1.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected float getRightInputPostDropoutScaling(AxonsContext axonsContext) {
    return 1f;
  }

  /**
   * Return the dropout mask for right hand side input. This is not yet supported, so we return
   * null.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the right hand side of these Axons
   */
  protected Matrix createRightInputDropoutMask(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    return null;
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
	  
    LOGGER.debug("Pushing right to left through Axons:" + rightNeuronsActivation.getFeatureCount() + ":" + rightNeuronsActivation.getExampleCount());
    if (rightNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }

    Matrix outputDropoutMask = null;

    Matrix inputDropoutMask = createRightInputDropoutMask(rightNeuronsActivation, axonsContext);

    Matrix previousInputDropoutMask = previousLeftToRightActivation == null ? null
        : previousLeftToRightActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null && isLeftInputDropoutSupported()) {
      LOGGER.debug("Transposing previous input dropout mask");
      outputDropoutMask = previousInputDropoutMask;
    }
    
    // TODO MON
    Matrix inputMatrix = null;
    if (inputDropoutMask != null) {
    	float postDropoutScaling = getRightInputPostDropoutScaling(axonsContext);
    	if (postDropoutScaling != 1) {
    		LOGGER.debug("Scaling post dropout right to left non-bias input");

    		inputMatrix = rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).mul(inputDropoutMask).asEditableMatrix()
    				.muli(postDropoutScaling);

    	} else {
    		inputMatrix = rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).mul(inputDropoutMask);
    	}
    } else {
    	//inputMatrix = rightNeuronsActivation.getActivations();
    }
    
    AxonsActivation axonsActivation = doPushRightToLeft(rightNeuronsActivation, previousInputDropoutMask, rightNeuronsActivation.getFeatureOrientation(), axonsContext);
    if (outputDropoutMask != null) {
      LOGGER.debug("Applying right to left output dropout mask");
      axonsActivation.applyOutputDropoutMask(outputDropoutMask);
      //outputDropoutMask.close();
    }
    
    LOGGER.debug("End Pushing right to left through Axons:" + axonsActivation.getOutput().getFeatureCount() + ":" + axonsActivation.getOutput().getExampleCount());

  
    return axonsActivation;
  }
}
