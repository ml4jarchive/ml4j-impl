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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class TrailingActivationFunctionDirectedComponentChainActivationImpl
		extends DefaultDirectedComponentChainActivationBase<TrailingActivationFunctionDirectedComponentChain> implements TrailingActivationFunctionDirectedComponentChainActivation {

	private DifferentiableActivationFunctionComponentActivation activationFunctionActivation;
	private DefaultDirectedComponentChainActivation precedingChainActivation;
	
	public TrailingActivationFunctionDirectedComponentChainActivationImpl(TrailingActivationFunctionDirectedComponentChain componentChain, DefaultDirectedComponentChainActivation precedingChainActivation,
			DifferentiableActivationFunctionComponentActivation activationFunctionActivation) {
		super(componentChain, Arrays.asList(precedingChainActivation, activationFunctionActivation), activationFunctionActivation.getOutput());
		this.activationFunctionActivation = activationFunctionActivation;
		this.precedingChainActivation = precedingChainActivation;
	}   

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient outerGradient) {
		
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = activationFunctionActivation.backPropagate(outerGradient);
		return precedingChainActivation.backPropagate(activationFunctionGradient);
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return getActivations().stream().flatMap(a -> a.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = activationFunctionActivation.backPropagate(outerGradient);
	
		DirectedComponentGradient<NeuronsActivation> result = precedingChainActivation.backPropagate(activationFunctionGradient);
	
		return result;
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		activationFunctionActivation.close(completedLifeCycleStage);
		precedingChainActivation.close(completedLifeCycleStage);
	}

}
