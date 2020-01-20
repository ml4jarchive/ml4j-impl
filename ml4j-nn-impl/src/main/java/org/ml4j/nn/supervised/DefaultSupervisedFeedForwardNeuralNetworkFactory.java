package org.ml4j.nn.supervised;

import java.util.List;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

public class DefaultSupervisedFeedForwardNeuralNetworkFactory implements SupervisedFeedForwardNeuralNetworkFactory {

	private DirectedComponentFactory directedComponentFactory;

	public DefaultSupervisedFeedForwardNeuralNetworkFactory(DirectedComponentFactory directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createSupervisedFeedForwardNeuralNetwork(
			List<DefaultChainableDirectedComponent<?, ?>> componentChain) {
		return new SupervisedFeedForwardNeuralNetworkImpl(directedComponentFactory, componentChain);
	}

}
