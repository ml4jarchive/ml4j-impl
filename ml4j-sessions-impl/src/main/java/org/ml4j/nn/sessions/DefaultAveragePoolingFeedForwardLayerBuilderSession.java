package org.ml4j.nn.sessions;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.axons.PoolingAxonsConfig;
import org.ml4j.nn.layers.AveragePoolingFeedForwardLayer;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.builders.AveragePoolingFeedForwardLayerPropertiesBuilder;

public class DefaultAveragePoolingFeedForwardLayerBuilderSession<C> extends DefaultDirected3DLayerBuilderSession<AveragePoolingFeedForwardLayer, C, Axons3DConfig, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<C>>
		implements FeedForward3DLayerBuilderSession<C, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<C>>, AveragePoolingFeedForwardLayerPropertiesBuilder<C> {
	
	public DefaultAveragePoolingFeedForwardLayerBuilderSession(String layerName,
			DirectedLayerFactory directedLayerFactory, Supplier<C> layerContainer,
			Consumer<AveragePoolingFeedForwardLayer> completedLayerConsumer) {
		super(layerName, directedLayerFactory, layerContainer, completedLayerConsumer);
	}

	@Override
	protected AveragePoolingFeedForwardLayer build(Axons3DConfig axons3DConfig) {
		return directedLayerFactory.createAveragePoolingFeedForwardLayer(layerName,
				PoolingAxonsConfig.create(axons3DConfig));
	}

	@Override
	protected AveragePoolingFeedForwardLayerPropertiesBuilder<C> getPropertiesBuilderInstance() {
		return this;
	}

	@Override
	protected Axons3DConfigBuilder createConfigBuilder() {
		return this.leftNeurons == null ? new Axons3DConfigBuilder() : new Axons3DConfigBuilder(leftNeurons);
	}

}
