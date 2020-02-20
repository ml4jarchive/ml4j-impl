package org.ml4j.nn.sessions;

import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;

public class DefaultLayeredSupervisedFeedForwardNeuralNetworkBuilderSession
		implements LayeredSupervisedFeedForwardNeuralNetworkBuilderSession {

	private String networkName;

	private DirectedLayerFactory directedLayerFactory;
	private DirectedComponentFactory directedComponentFactory;
	private LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory;

	private List<FeedForwardLayer<?, ?>> layers;

	public DefaultLayeredSupervisedFeedForwardNeuralNetworkBuilderSession(
			DirectedComponentFactory directedComponentFactory, DirectedLayerFactory directedLayerFactory,
			LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory, 
			List<FeedForwardLayer<?, ?>> layers, String networkName) {
		this.directedComponentFactory = directedComponentFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.layeredNeuralNetworkFactory = layeredNeuralNetworkFactory;
		this.networkName = networkName;
		this.layers = layers;
	}

	@Override
	public List<FeedForwardLayer<?, ?>> getComponents() {
		return layers;
	}

	@Override
	public LayeredSupervisedFeedForwardNeuralNetwork build() {
		if (layeredNeuralNetworkFactory != null) {
			return layeredNeuralNetworkFactory.createLayeredSupervisedFeedForwardNeuralNetwork(networkName, layers);
		} else {
			throw new IllegalStateException(
					"No neural network factory available for LayeredSupervisedFeedForwardNeuralNetworks");
		}
	}

	@Override
	public DirectedComponentFactory getDirectedComponentFactory() {
		return directedComponentFactory;
	}

	@Override
	public DirectedLayerFactory getDirectedLayerFactory() {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return directedLayerFactory;
	}

	@Override
	public FullyConnectedFeedForwardLayerBuilderSession<LayeredSupervisedFeedForwardNeuralNetworkBuilderSession> withFullyConnectedLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultFullyConnectedFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this,
				getComponents()::add);
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>, L extends FeedForwardLayer<A, L>> LayeredSupervisedFeedForwardNeuralNetworkBuilderSession withLayer(
			L layer) {
		getComponents().add(layer);
		return new DefaultLayeredSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactory, directedLayerFactory, layeredNeuralNetworkFactory,  getComponents(), networkName);
	}
}
