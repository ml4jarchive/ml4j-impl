package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;

public class TrailingActivationFunctionDirectedComponentChainActivationImpl
		extends DefaultDirectedComponentChainActivationBase implements TrailingActivationFunctionDirectedComponentChainActivation {

	private DifferentiableActivationFunctionActivation activationFunctionActivation;
	private DefaultDirectedComponentChainActivation precedingChainActivation;
	
	public TrailingActivationFunctionDirectedComponentChainActivationImpl(DefaultDirectedComponentChainActivation precedingChainActivation,
			DifferentiableActivationFunctionActivation activationFunctionActivation) {
		super(activationFunctionActivation.getOutput());
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
	public List<DefaultChainableDirectedComponentActivation> getActivations() {
		List<DefaultChainableDirectedComponentActivation> activations = new ArrayList<>();
		activations.addAll(precedingChainActivation.getActivations());
		activations.add(activationFunctionActivation);
		return activations;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = activationFunctionActivation.backPropagate(outerGradient);
		return precedingChainActivation.backPropagate(activationFunctionGradient);
	}

}
