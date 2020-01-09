package org.ml4j.nn.layers;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.onetoone.DirectedComponentChainActivationImpl;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedLayerChainActivationImpl
		extends DirectedComponentChainActivationImpl<NeuronsActivation, DirectedLayerActivation> implements DirectedLayerChainActivation {

	private DirectedLayerActivation finalLayerActivation;
	private DirectedLayerChainActivation precedingChainActivation;
	
	public DirectedLayerChainActivationImpl(List<DirectedLayerActivation> directedLayerActivations) {
		super(getActivations(directedLayerActivations), getFinalActivation(directedLayerActivations).getOutput());
		this.finalLayerActivation = getFinalActivation(directedLayerActivations);
		if (this.getActivations().size() > 1) { 
			this.precedingChainActivation = new DirectedLayerChainActivationImpl(getActivations(directedLayerActivations).subList(0, directedLayerActivations.size() - 1));
		}
	}   
	
	private static DirectedLayerActivation getFinalActivation(List<DirectedLayerActivation> directedLayerActivations) {
		if (directedLayerActivations.isEmpty()) {
			throw new IllegalArgumentException("At least one instance of DirectedLayerActivation is required in a DirectedLayerChainActivation");
		}
		return directedLayerActivations.get(directedLayerActivations.size() - 1);
	}

	private static List<DirectedLayerActivation> getActivations(List<DirectedLayerActivation> directedSynapsesActivations) {
		if (directedSynapsesActivations.isEmpty()) {
			throw new IllegalArgumentException("At least one instance of DirectedLayerActivation is required in a DirectedLayerChainActivation");
		}
		return directedSynapsesActivations;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient outerGradient) {
		
		DirectedComponentGradient<NeuronsActivation> activationFunctionGradient = finalLayerActivation.backPropagate(outerGradient);
		if (precedingChainActivation != null) {
			return precedingChainActivation.backPropagate(activationFunctionGradient);
		} else {
			return activationFunctionGradient;
		}
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		finalLayerActivation.close(completedLifeCycleStage);
		precedingChainActivation.close(completedLifeCycleStage);
	}


}
