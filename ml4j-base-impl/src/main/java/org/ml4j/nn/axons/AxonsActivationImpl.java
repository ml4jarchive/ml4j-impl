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

import java.util.function.Supplier;

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
  private AxonsDropoutMask dropoutMask;
  private NeuronsActivation outputActivations;
  private Supplier<NeuronsActivation> postDropoutInput;
  
  /**
   * @param inputDropoutMask Any input dropout mask
   * @param postDropoutInput The post dropout input
   * @param outputActivations The output.
   */
  public AxonsActivationImpl(Axons<?, ?, ?> axons, AxonsDropoutMask dropoutMask, 
		  Supplier<NeuronsActivation> postDropoutInput, NeuronsActivation outputActivations, Neurons leftNeurons, Neurons rightNeurons) {
    this.outputActivations = outputActivations;
    this.dropoutMask = dropoutMask;
    this.postDropoutInput = postDropoutInput;
    //this.postDropoutInput.setImmutable(true);
	if (outputActivations.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		throw new IllegalStateException("Currently only ROWS_SPAN_FEATURE_SET orientation supported");
	}
	
    this.axons = axons;
	
  }
  
  @Override
  public NeuronsActivation getPostDropoutOutput() {
    return outputActivations;
  }

  @Override
  public AxonsDropoutMask getDropoutMask() {
    return dropoutMask;
  }

  @Override
  public Supplier<NeuronsActivation> getPostDropoutInput() {
    return postDropoutInput;
  }
  
  @Override
  public Axons<?, ?, ?> getAxons() {
    return axons;
  }

}
