package org.ml4j.nn.sessions;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.builders.AveragePoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.MaxPoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

public class DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession
		implements SupervisedFeedForwardNeuralNetwork3DBuilderSession{

	private List<DefaultChainableDirectedComponent<?, ?>> components;
	private String networkName;
	
	private DirectedLayerFactory directedLayerFactory;
	private SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory;
	private DirectedComponentFactory directedComponentFactory;
	private LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory;

	
	public DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession(DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory, 
			LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory, String networkName) {
		this.components = new ArrayList<>();
		this.directedLayerFactory = directedLayerFactory;
		this.neuralNetworkFactory = neuralNetworkFactory;
		this.layeredNeuralNetworkFactory = layeredNeuralNetworkFactory;
		this.networkName = networkName;
		this.directedComponentFactory = directedComponentFactory;
	}
	
	public List<DefaultChainableDirectedComponent<?, ?>> getLayers() {
		return components;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork build() {

		if (neuralNetworkFactory != null) {
			return neuralNetworkFactory.createSupervisedFeedForwardNeuralNetwork(networkName, components);
		} else {
			throw new IllegalStateException("No neural network factory available for SupervisedFeedForwardNeuralNetworks");
		}		
	}
		
	@Override
	public LayeredSupervisedFeedForwardNeuralNetwork buildLayeredNeuralNetwork() {
		// TODO - refactor this and use type-specific lists.
		boolean notFeedForwardLayer = false;
		List<FeedForwardLayer<?, ?>> layers = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> component : components) {
			if (!(component instanceof FeedForwardLayer)) {
				notFeedForwardLayer = true;
			} else {
				layers.add((FeedForwardLayer<?, ?>)component);
			}
		}
		if (!notFeedForwardLayer) {
			if (layeredNeuralNetworkFactory != null) {
				return layeredNeuralNetworkFactory.createLayeredSupervisedFeedForwardNeuralNetwork(networkName, layers);
			} else {
				throw new IllegalStateException("No neural network factory available for LayeredSupervisedFeedForwardNeuralNetworks");
			}
		} else {
			throw new IllegalStateException("Not all the components in the network are FeedForwardLayers");
		}
		
	}

	@Override
	public DirectedComponentFactory getDirectedComponentFactory() {
		return directedComponentFactory;
	}

	@Override
	public ConvolutionalFeedForwardLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession> withConvolutionalLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultConvolutionalFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, components::add);
	}

	@Override
	public DirectedLayerFactory getDirectedLayerFactory() {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return directedLayerFactory;
	}

	@Override
	public FeedForward3DLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<SupervisedFeedForwardNeuralNetwork3DBuilderSession>> withAveragePoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultAveragePoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, components::add);
	}
	
	@Override
	public FeedForward3DLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<SupervisedFeedForwardNeuralNetwork3DBuilderSession>> withMaxPoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultMaxPoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, components::add);
	}

	@Override
	public FullyConnectedFeedForwardLayerBuilderSession<SupervisedFeedForwardNeuralNetworkBuilderSession> withFullyConnectedLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultFullyConnectedFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, components::add);
	}

}
