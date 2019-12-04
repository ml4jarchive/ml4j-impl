package org.ml4j.nn.supervised;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultSupervisedFeedForwardNeuralNetworkFactory implements SupervisedFeedForwardNeuralNetworkFactory {

	@Override
	public SupervisedFeedForwardNeuralNetwork createSupervisedFeedForwardNeuralNetwork(
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> componentChain) {
		return new SupervisedFeedForwardNeuralNetworkImpl(componentChain);
	}

}
