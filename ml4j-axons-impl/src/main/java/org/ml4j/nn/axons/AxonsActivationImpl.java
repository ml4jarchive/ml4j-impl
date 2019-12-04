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

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * Encapsulates the artifacts produced when pushing NeuronsActivations
 * through an Axons instance.
 * 
 * @author Michael Lavelle
 */
public class AxonsActivationImpl implements AxonsActivation {

  private Axons<?, ?, ?> axons;
  private Matrix inputDropoutMask;
  private NeuronsActivation outputActivations;
  private NeuronsActivation postDropoutInput;
  private boolean rightToLeft;
  
  /**
   * @param inputDropoutMask Any input dropout mask
   * @param postDropoutInput The post dropout input
   * @param outputActivations The output.
   */
  public AxonsActivationImpl(Axons<?, ?, ?> axons, Matrix inputDropoutMask, 
		  NeuronsActivation postDropoutInput, NeuronsActivation outputActivations, Neurons leftNeurons, Neurons rightNeurons, boolean rightToLeft) {
    this.outputActivations = outputActivations;
    this.rightToLeft = rightToLeft;
    this.inputDropoutMask = inputDropoutMask;
    this.postDropoutInput = postDropoutInput;
    this.postDropoutInput.setImmutable(true);

	if (postDropoutInput.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		throw new IllegalStateException("Currently only ROWS_SPAN_FEATURE_SET orientation supported");
	}
	if (outputActivations.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		throw new IllegalStateException("Currently only ROWS_SPAN_FEATURE_SET orientation supported");
	}
	
    
    if (rightToLeft)  {
    	  //if (this.postDropoutInput.getActivations().getRows() != rightNeurons.getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows:" + postDropoutInput.getActivations().getRows() + ":" + rightNeurons.getNeuronCountExcludingBias());
    	    //}
    	    if (inputDropoutMask != null && this.inputDropoutMask.getRows() != rightNeurons.getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows");
    	    }
    	    if (outputActivations.getFeatureCount() != axons.getLeftNeurons().getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows:" + outputActivations.getActivations().getRows());
    	    }
    	    //if (outputActivations.getActivations().getColumns() != postDropoutInput.getActivations().getColumns()) {
    	    	//throw new IllegalArgumentException("Incorrect number of columns");
    	    //}
    } else {
    	  //if (this.postDropoutInput.getActivations().getRows() != leftNeurons.getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows:" + postDropoutInput.getActivations().getRows() + ":" + leftNeurons.getNeuronCountExcludingBias());
    	    //}
    	    if (inputDropoutMask != null && this.inputDropoutMask.getRows() != leftNeurons.getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows");
    	    }
    	    if (outputActivations.getFeatureCount() != axons.getRightNeurons().getNeuronCountExcludingBias()) {
    	    	//throw new IllegalArgumentException("Incorrect number of rows:" + outputActivations.getActivations().getRows() + ":" + axons.getRightNeurons().getNeuronCountExcludingBias());
    	    }
    	    //if (outputActivations.getActivations().getColumns() != postDropoutInput.getActivations().getColumns()) {
    	    	//throw new IllegalArgumentException("Incorrect number of columns:" + outputActivations.getActivations().getColumns() + ":" + postDropoutInput.getActivations().getColumns());
    	    //}
    }
  
    this.axons = axons;
	
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivations;
  }

  @Override
  public Matrix getInputDropoutMask() {
    return inputDropoutMask;
  }

  @Override
  public NeuronsActivation getPostDropoutInput() {
    return postDropoutInput;
  }
  
  @Override
  public Axons<?, ?, ?> getAxons() {
    return axons;
  }

  @Override
  public void applyOutputDropoutMask(Matrix outputDropoutMask) {
// TODO
	  	this.outputActivations.getActivations(null).asEditableMatrix().muli(outputDropoutMask);
  }
}
