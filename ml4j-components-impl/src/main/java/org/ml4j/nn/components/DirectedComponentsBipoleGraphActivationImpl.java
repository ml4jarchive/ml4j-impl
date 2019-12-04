package org.ml4j.nn.components;

import java.util.List;
import java.util.stream.Collectors;

public class DirectedComponentsBipoleGraphActivationImpl<I, A extends ChainableDirectedComponentActivation<I>>
		implements DirectedComponentsBipoleGraphActivation<I> {

	private I output;

	protected OneToManyDirectedComponentActivation<I> inputLinkActivation;
	protected ManyToOneDirectedComponentActivation<I> outputLinkActivation;
	protected DirectedComponentBatchActivation<I, A> edgesActivation;

	public DirectedComponentsBipoleGraphActivationImpl(OneToManyDirectedComponentActivation<I> inputLinkActivation,
			DirectedComponentBatchActivation<I, A> edgesActivation,
			ManyToOneDirectedComponentActivation<I> outputLinkActivation) {
		this.inputLinkActivation = inputLinkActivation;
		this.outputLinkActivation = outputLinkActivation;
		this.edgesActivation = edgesActivation;
		this.output = outputLinkActivation.getOutput();
	}

	@Override
	public DirectedComponentGradient<I> backPropagate(DirectedComponentGradient<I> outerGradient) {

		DirectedComponentGradient<List<I>> manyToOneActivation = outputLinkActivation.backPropagate(outerGradient);
		DirectedComponentGradient<List<I>> edgesGradients = edgesActivation.backPropagate(manyToOneActivation);
		return inputLinkActivation.backPropagate(edgesGradients);
	}

	public DirectedComponentBatchActivation<I, A> getEdges() {
		return edgesActivation;
	}

	@Override
	public I getOutput() {
		return output;
	}

	@Override
	public List<ChainableDirectedComponentActivation<I>> decompose() {
		return edgesActivation.getActivations().stream().flatMap(a -> a.decompose().stream())
				.collect(Collectors.toList());
	}

}
