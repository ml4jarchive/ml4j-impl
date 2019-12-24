package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;

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
