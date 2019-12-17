package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class TrailingActivationFunctionDirectedComponentChainActivationImpl
		extends DefaultDirectedComponentChainActivationImpl implements TrailingActivationFunctionDirectedComponentChainActivation {

	private DifferentiableActivationFunctionActivation activationFunctionActivation;
	private DefaultDirectedComponentChainActivation precedingChainActivation;
	
	public TrailingActivationFunctionDirectedComponentChainActivationImpl(DefaultDirectedComponentChainActivation precedingChainActivation,
			DifferentiableActivationFunctionActivation activationFunctionActivation) {
		super(getActivations(precedingChainActivation, activationFunctionActivation), activationFunctionActivation.getOutput());
		this.activationFunctionActivation = activationFunctionActivation;
		this.precedingChainActivation = precedingChainActivation;
	}   

	private static List<DefaultChainableDirectedComponentActivation> getActivations(
			DefaultDirectedComponentChainActivation precedingChainActivation,
			DifferentiableActivationFunctionActivation activationFunctionActivation) {
		List<DefaultChainableDirectedComponentActivation> activations = new ArrayList<>();
		activations.addAll(precedingChainActivation.getActivations());
		activations.add(activationFunctionActivation);
		return activations;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient outerGradient) {
		
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = activationFunctionActivation.backPropagate(outerGradient);
		return precedingChainActivation.backPropagate(activationFunctionGradient);
	}

}
