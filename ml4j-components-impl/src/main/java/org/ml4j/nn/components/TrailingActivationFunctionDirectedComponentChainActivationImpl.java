package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class TrailingActivationFunctionDirectedComponentChainActivationImpl
		extends DirectedComponentChainActivationImpl<NeuronsActivation, ChainableDirectedComponentActivation<NeuronsActivation>> implements TrailingActivationFunctionDirectedComponentChainActivation {

	private DifferentiableActivationFunctionActivation activationFunctionActivation;
	private DirectedComponentChainActivation<NeuronsActivation, ChainableDirectedComponentActivation<NeuronsActivation>> precedingChainActivation;
	
	public TrailingActivationFunctionDirectedComponentChainActivationImpl(DirectedComponentChainActivation<NeuronsActivation, ChainableDirectedComponentActivation<NeuronsActivation>> precedingChainActivation,
			DifferentiableActivationFunctionActivation activationFunctionActivation) {
		super(getActivations(precedingChainActivation, activationFunctionActivation), activationFunctionActivation.getOutput());
		this.activationFunctionActivation = activationFunctionActivation;
		this.precedingChainActivation = precedingChainActivation;
	}   

	private static List<ChainableDirectedComponentActivation<NeuronsActivation>> getActivations(
			DirectedComponentChainActivation<NeuronsActivation, ChainableDirectedComponentActivation<NeuronsActivation>> precedingChainActivation,
			DifferentiableActivationFunctionActivation activationFunctionActivation) {
		List<ChainableDirectedComponentActivation<NeuronsActivation>> activations = new ArrayList<>();
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
