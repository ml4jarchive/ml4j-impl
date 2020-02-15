package org.ml4j.nn.sessions;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.Axons3DConfigBuilderBase;
import org.ml4j.nn.axons.Axons3DConfigPopulator;
import org.ml4j.nn.layers.DirectedLayer;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.builders.FeedForward3DLayerPropertiesBuilder;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class DefaultDirected3DLayerBuilderSession<L extends DirectedLayer<?, L>, C, D extends Axons3DConfig, A extends Axons3DConfigBuilderBase<D, A>, B extends FeedForward3DLayerPropertiesBuilder<C, A>>
		implements FeedForward3DLayerBuilderSession<C, A, B> {

	protected final DirectedLayerFactory directedLayerFactory;
	protected Supplier<C> layerContainer;
	protected final String layerName;
	protected final Consumer<L> completedLayerConsumer;
	protected Neurons3D leftNeurons;
	protected Axons3DConfigPopulator<A> configPopulator;


	public DefaultDirected3DLayerBuilderSession(String layerName, 
			DirectedLayerFactory directedLayerFactory, Supplier<C> layerContainer, Consumer<L> completedLayerConsumer) {
		this.layerContainer = layerContainer;
		this.directedLayerFactory = directedLayerFactory;
		this.layerName = layerName;
		this.completedLayerConsumer = completedLayerConsumer;
		this.configPopulator = new PrototypeAxons3DConfigPopulatorImpl<>();
	}
	
	@Override
	public B withInputNeurons(Neurons3D leftNeurons) {
		this.leftNeurons = leftNeurons;
		return getPropertiesBuilderInstance();
	}
	
	protected abstract B getPropertiesBuilderInstance();
	
	protected abstract A createConfigBuilder();
	
	protected abstract L build(D axons3DConfig);

	public DefaultDirected3DLayerBuilderSession<L, C, D, A, B> withLayerContainer(Supplier<C> layerContainer) {
		this.layerContainer = layerContainer;
		return this;
	}

	protected void addCompletedLayer(L completedLayer) {
		completedLayerConsumer.accept(completedLayer);
	}


	//@Override
	public C withOutputNeurons(Neurons3D outputNeurons) {

		A axons3DConfigBuilder = createConfigBuilder();

		axons3DConfigBuilder.withOutputNeurons(outputNeurons);

		D axons3DConfig = axons3DConfigBuilder.build(configPopulator);
		
		L layer = build(axons3DConfig);
		
		addCompletedLayer(layer);

		return layerContainer.get();
	}

	@Override
	public C withConfig(Consumer<A> configConfigurer) {

		A axons3DConfigBuilder = createConfigBuilder();
		
		configConfigurer.accept(axons3DConfigBuilder);
				
		D axons3DConfig = axons3DConfigBuilder.build(configPopulator);
	
		L layer = build(axons3DConfig);
		
		addCompletedLayer(layer);

		return layerContainer.get();
	}

}
