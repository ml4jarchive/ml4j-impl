/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.generic.DirectedComponentChainActivation;

public class DirectedComponentChainActivationImpl<I, A extends ChainableDirectedComponentActivation<I>> implements DirectedComponentChainActivation<I, A> {

	private I output;
	protected List<A> activations;
	
	public DirectedComponentChainActivationImpl(List<A> activations, I output) {
		this.activations = activations;
		this.output = output;
	}
	
	@Override
	public DirectedComponentGradient<I> backPropagate(DirectedComponentGradient<I> outerGradient) {
		List<ChainableDirectedComponentActivation<I>> reversedSynapseActivations =
			        new ArrayList<>();
			    reversedSynapseActivations.addAll(activations);
			    Collections.reverse(reversedSynapseActivations);
			    return backPropagateAndAddToSynapseGradientList(outerGradient,
			        reversedSynapseActivations);
	}
	
	private DirectedComponentGradient<I> backPropagateAndAddToSynapseGradientList(
		      DirectedComponentGradient<I> outerSynapsesGradient,
		      List<ChainableDirectedComponentActivation<I>> activationsToBackPropagateThrough) {

			List<Supplier<AxonsGradient>> totalTrainableAxonsGradients = new ArrayList<>();
			totalTrainableAxonsGradients.addAll(outerSynapsesGradient.getTotalTrainableAxonsGradients());
			
		    DirectedComponentGradient<I> finalGrad = outerSynapsesGradient;
		    DirectedComponentGradient<I> synapsesGradient = outerSynapsesGradient;
		    List<Supplier<AxonsGradient>> finalTotalTrainableAxonsGradients = outerSynapsesGradient.getTotalTrainableAxonsGradients();
		    List<DirectedComponentGradient<I>> componentGradients = new ArrayList<>();
		    for (ChainableDirectedComponentActivation<I> synapsesActivation : activationsToBackPropagateThrough) {
		     
		      componentGradients.add(synapsesGradient);
		      synapsesGradient = 
		          synapsesActivation.backPropagate(synapsesGradient);
		   
		      finalTotalTrainableAxonsGradients = synapsesGradient.getTotalTrainableAxonsGradients();
		      finalGrad = synapsesGradient;
		    }
		    return new DirectedComponentGradientImpl<>(finalTotalTrainableAxonsGradients, finalGrad.getOutput());
		  }


	@Override
	public List<A> getActivations() {
		return activations;
	}

	@Override
	public I getOutput() {
		return output;
	}

	@Override
	public List<? extends ChainableDirectedComponentActivation<I>> decompose() {
		List<ChainableDirectedComponentActivation<I>> acts = new ArrayList<>();
		acts.addAll(activations);
		return acts;
	}
}
