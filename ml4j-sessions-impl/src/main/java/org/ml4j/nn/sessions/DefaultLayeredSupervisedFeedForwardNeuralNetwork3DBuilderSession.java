package org.ml4j.nn.sessions;

import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.builders.AveragePoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.MaxPoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;

public class DefaultLayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession 
		implements LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession{

	private String networkName;
	private List<FeedForwardLayer<?, ?>> layers;
	
	private DirectedLayerFactory directedLayerFactory;
	private DirectedComponentFactory directedComponentFactory;
	private LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory;
	
	public DefaultLayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession(
			DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			LayeredSupervisedFeedForwardNeuralNetworkFactory layeredNeuralNetworkFactory, List<FeedForwardLayer<?, ?>> layers, String networkName) {
		this.directedComponentFactory = directedComponentFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.layeredNeuralNetworkFactory = layeredNeuralNetworkFactory;
		this.networkName = networkName;
		this.layers = layers;
	}
	
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
	public ConvolutionalFeedForwardLayerBuilderSession<LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession> withConvolutionalLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultConvolutionalFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}

	@Override
	public DirectedLayerFactory getDirectedLayerFactory() {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return directedLayerFactory;
	}

	@Override
	public FeedForward3DLayerBuilderSession<LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession>> withAveragePoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultAveragePoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}
	
	@Override
	public FeedForward3DLayerBuilderSession<LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession>> withMaxPoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultMaxPoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}

	@Override
	public FullyConnectedFeedForwardLayerBuilderSession<LayeredSupervisedFeedForwardNeuralNetworkBuilderSession> withFullyConnectedLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultFullyConnectedFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>, L extends FeedForwardLayer<A, L>> LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession with3DLayer(
			L layer) {
		this.getComponents().add(layer);
		return this;
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>, L extends FeedForwardLayer<A, L>> LayeredSupervisedFeedForwardNeuralNetworkBuilderSession withLayer(
			L layer) {
		this.getComponents().add(layer);
		return new DefaultLayeredSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactory, directedLayerFactory, layeredNeuralNetworkFactory, getComponents(), networkName);
	}

}
