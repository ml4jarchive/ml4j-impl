package org.ml4j.nn.supervised;

import java.util.List;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;

public class DefaultSupervisedFeedForwardNeuralNetworkFactory implements SupervisedFeedForwardNeuralNetworkFactory {

	@Override
	public SupervisedFeedForwardNeuralNetwork createSupervisedFeedForwardNeuralNetwork(
			List<DefaultChainableDirectedComponent<?, ?>> componentChain) {
		return new SupervisedFeedForwardNeuralNetworkImpl(componentChain);
	}

}
