package org.ml4j.nn.sessions;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.layers.AveragePoolingFeedForwardLayer;
import org.ml4j.nn.layers.ConvolutionalFeedForwardLayer;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayer;
import org.ml4j.nn.layers.MaxPoolingFeedForwardLayer;
import org.ml4j.nn.layers.builders.AveragePoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.MaxPoolingFeedForwardLayerPropertiesBuilder;

public class DefaultDirectedLayerBuilderSession implements DirectedLayerBuilderSession {

	private DirectedLayerFactory directedLayerFactory;

	public DefaultDirectedLayerBuilderSession(DirectedLayerFactory directedLayerFactory) {
		this.directedLayerFactory = directedLayerFactory;
	}

	@Override
	public DirectedLayerFactory getDirectedLayerFactory() {
		return directedLayerFactory;
	}

	@Override
	public FullyConnectedFeedForwardLayerBuilderSession<FullyConnectedFeedForwardLayer> buildFullyConnectedLayer(
			String layerName) {
		List<FullyConnectedFeedForwardLayer> layers = new ArrayList<>();
		return new DefaultFullyConnectedFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory,
				() -> Optional.ofNullable(layers.size() == 1 ? layers.get(0) : null)
						.orElseThrow(() -> new IllegalStateException("No layer has been built")),
				layers::add);
	}

	@Override
	public FeedForward3DLayerBuilderSession<AveragePoolingFeedForwardLayer, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<AveragePoolingFeedForwardLayer>> buildAveragePoolingLayer(
			String layerName) {
		List<AveragePoolingFeedForwardLayer> layers = new ArrayList<>();
		return new DefaultAveragePoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory,
				() -> Optional.ofNullable(layers.size() == 1 ? layers.get(0) : null)
						.orElseThrow(() -> new IllegalStateException("No layer has been built")),
				layers::add);

	}

	@Override
	public FeedForward3DLayerBuilderSession<MaxPoolingFeedForwardLayer, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<MaxPoolingFeedForwardLayer>> buildMaxPoolingLayer(
			String layerName) {
		List<MaxPoolingFeedForwardLayer> layers = new ArrayList<>();
		return new DefaultMaxPoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory,
				() -> Optional.ofNullable(layers.size() == 1 ? layers.get(0) : null)
						.orElseThrow(() -> new IllegalStateException("No layer has been built")),
				layers::add);
	}

	@Override
	public ConvolutionalFeedForwardLayerBuilderSession<ConvolutionalFeedForwardLayer> buildConvolutionalLayer(
			String layerName) {
		List<ConvolutionalFeedForwardLayer> layers = new ArrayList<>();
		return new DefaultConvolutionalFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory,
				() -> Optional.ofNullable(layers.size() == 1 ? layers.get(0) : null)
						.orElseThrow(() -> new IllegalStateException("No layer has been built")),
				layers::add);
	}

}
