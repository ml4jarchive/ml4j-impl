package org.ml4j.nn.sessions;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.MaxPoolingFeedForwardLayer;
import org.ml4j.nn.layers.builders.MaxPoolingFeedForwardLayerPropertiesBuilder;

public class DefaultMaxPoolingFeedForwardLayerBuilderSession<C> extends DefaultDirected3DLayerBuilderSession<MaxPoolingFeedForwardLayer, C, Axons3DConfig, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<C>>
		implements FeedForward3DLayerBuilderSession<C, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<C>>, MaxPoolingFeedForwardLayerPropertiesBuilder<C> {
	
	private boolean scaleOutputs;
	
	public DefaultMaxPoolingFeedForwardLayerBuilderSession(String layerName,
			DirectedLayerFactory directedLayerFactory, Supplier<C> layerContainer,
			Consumer<MaxPoolingFeedForwardLayer> completedLayerConsumer) {
		super(layerName, directedLayerFactory, layerContainer, completedLayerConsumer);
	}

	@Override
	protected MaxPoolingFeedForwardLayer build(Axons3DConfig axons3DConfig) {
		return directedLayerFactory.createMaxPoolingFeedForwardLayer(layerName,
				PoolingAxonsConfig.create(axons3DConfig), scaleOutputs);
	}

	@Override
	protected MaxPoolingFeedForwardLayerPropertiesBuilder<C> getPropertiesBuilderInstance() {
		return this;
	}

	@Override
	protected Axons3DConfigBuilder createConfigBuilder() {
		return this.leftNeurons == null ? new Axons3DConfigBuilder() : new Axons3DConfigBuilder(leftNeurons);
	}

	@Override
	public MaxPoolingFeedForwardLayerPropertiesBuilder<C> withScaleOutputs() {
		this.scaleOutputs = true;
		return this;
	}

}
