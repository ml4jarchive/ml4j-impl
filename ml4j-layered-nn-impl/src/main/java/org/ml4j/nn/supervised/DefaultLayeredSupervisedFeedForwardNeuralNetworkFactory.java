package org.ml4j.nn.supervised;

import java.util.List;

import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.layers.FeedForwardLayer;

public class DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory implements LayeredSupervisedFeedForwardNeuralNetworkFactory {

	private DirectedComponentFactory directedComponentFactory;

	public DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory(DirectedComponentFactory directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}

	@Override
	public LayeredSupervisedFeedForwardNeuralNetwork createLayeredSupervisedFeedForwardNeuralNetwork(String name,
			List<FeedForwardLayer<?, ?>> layerChain) {
		return new LayeredSupervisedFeedForwardNeuralNetworkImpl(name, directedComponentFactory, layerChain);
	}
}
