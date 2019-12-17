package org.ml4j.nn.synapses;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentChainActivationImpl;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesChainActivationImpl
		extends DirectedComponentChainActivationImpl<NeuronsActivation, DirectedSynapsesActivation> implements DirectedSynapsesChainActivation {

	private DirectedSynapsesActivation finalSynapsesActivation;
	private DirectedSynapsesChainActivation precedingChainActivation;
	
	public DirectedSynapsesChainActivationImpl(List<DirectedSynapsesActivation> directedSynapsesActivations, NeuronsActivation output) {
		super(getActivations(directedSynapsesActivations), output);
		this.finalSynapsesActivation = getFinalActivation(directedSynapsesActivations);
		if (this.getActivations().size() > 1) { 
			this.precedingChainActivation = new DirectedSynapsesChainActivationImpl(getActivations(directedSynapsesActivations).subList(0, directedSynapsesActivations.size() - 1), output);
		}
	}   
	
	private static DirectedSynapsesActivation getFinalActivation(List<DirectedSynapsesActivation> directedSynapsesActivations) {
		if (directedSynapsesActivations.isEmpty()) {
			throw new IllegalArgumentException("At least one instance of DirectedSynapsesActivation is required in a DirectedSynapsesChainActivation");
		}
		return directedSynapsesActivations.get(directedSynapsesActivations.size() - 1);
	}

	private static List<DirectedSynapsesActivation> getActivations(List<DirectedSynapsesActivation> directedSynapsesActivations) {
		if (directedSynapsesActivations.isEmpty()) {
			throw new IllegalArgumentException("At least one instance of DirectedSynapsesActivation is required in a DirectedSynapsesChainActivation");
		}
		return directedSynapsesActivations;
	}


	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient outerGradient) {
		
		if (finalSynapsesActivation == null) {
			throw new IllegalArgumentException("At least one instance of DirectedSynapsesActivation is required in a DirectedSynapsesChainActivation");
		}
		
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = finalSynapsesActivation.backPropagate(outerGradient);
		if (precedingChainActivation != null) {
			return precedingChainActivation.backPropagate(activationFunctionGradient);
		} else {
			return activationFunctionGradient;
		}
	}
}
