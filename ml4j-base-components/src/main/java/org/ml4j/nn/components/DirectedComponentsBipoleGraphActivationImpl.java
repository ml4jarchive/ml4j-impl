package org.ml4j.nn.components;

import java.util.List;
import java.util.stream.Collectors;

public class DirectedComponentsBipoleGraphActivationImpl<I, A extends ChainableDirectedComponentActivation<I>>
		implements DirectedComponentBipoleGraphActivation<I> {

	private I output;

	protected GenericOneToManyDirectedComponentActivation<I> inputLinkActivation;
	protected GenericManyToOneDirectedComponentActivation<I> outputLinkActivation;
	protected DirectedComponentBatchActivation<I, A> edgesActivation;

	public DirectedComponentsBipoleGraphActivationImpl(GenericOneToManyDirectedComponentActivation<I> inputLinkActivation,
			DirectedComponentBatchActivation<I, A> edgesActivation,
			GenericManyToOneDirectedComponentActivation<I> outputLinkActivation) {
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
	public List<? extends ChainableDirectedComponentActivation<I>> decompose() {
		return edgesActivation.getActivations().stream().flatMap(a -> a.decompose().stream())
				.collect(Collectors.toList());
	}

}
