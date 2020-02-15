package org.ml4j.nn.sessions;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.AxonsConfigBuilderBase;
import org.ml4j.nn.layers.DirectedLayer;
import org.ml4j.nn.layers.DirectedLayerFactory;

public abstract class DefaultDirectedNon3DLayerBuilderSession<L extends DirectedLayer<?, L>, C, D extends AxonsConfig<?, ?>, A extends AxonsConfigBuilderBase<D, A>, B> {

	protected final DirectedLayerFactory directedLayerFactory;
	protected Supplier<C> layerContainer;
	protected final String layerName;
	protected final Consumer<L> completedLayerConsumer;

	public DefaultDirectedNon3DLayerBuilderSession(String layerName, DirectedLayerFactory directedLayerFactory,
			Supplier<C> layerContainer, Consumer<L> completedLayerConsumer) {
		this.layerContainer = layerContainer;
		this.directedLayerFactory = directedLayerFactory;
		this.layerName = layerName;
		this.completedLayerConsumer = completedLayerConsumer;
	}

	protected B withLayerContainer(Supplier<C> layerContainer) {
		this.layerContainer = layerContainer;
		return this.getPropertiesBuilderInstance();
	}

	protected abstract B getPropertiesBuilderInstance();

	protected abstract A createConfigBuilder();

	protected abstract L build(D axons3DConfig);

	protected void addCompletedLayer(L completedLayer) {
		completedLayerConsumer.accept(completedLayer);
	}

}
