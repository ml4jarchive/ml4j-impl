package org.ml4j.nn.sessions;

import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfigBuilder;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.definitions.Component3Dto3DGraphDefinition;
import org.ml4j.nn.definitions.Component3DtoNon3DGraphDefinition;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.builders.AveragePoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.MaxPoolingFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

public class DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession extends InitialComponents3DGraphBuilderImpl<DefaultChainableDirectedComponent<?, ?>>
		implements SupervisedFeedForwardNeuralNetwork3DBuilderSession{

	private String networkName;
	
	private DirectedLayerFactory directedLayerFactory;
	private SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory;
	private DirectedComponentFactory directedComponentFactoryReference;
	private Neurons3D initialNeurons;
	
	public DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession(
			DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory, String networkName,
			DirectedComponentsContext directedComponentsContext, List<DefaultChainableDirectedComponent<?, ?>> components, Neurons3D currentNeurons) {
		super(directedComponentFactory, directedComponentsContext, currentNeurons);
		this.directedComponentFactoryReference = directedComponentFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.neuralNetworkFactory = neuralNetworkFactory;
		this.networkName = networkName;
		this.initialNeurons = currentNeurons;
		this.getComponents().addAll(components);
	}
	
	@Override
	public String getNetworkName() {
		return networkName;
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetworkFactory getNeuralNetworkFactory() {
		return neuralNetworkFactory;
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork build() {

		if (neuralNetworkFactory != null) {
			return neuralNetworkFactory.createSupervisedFeedForwardNeuralNetwork(networkName, getComponents());
		} else {
			throw new IllegalStateException("No neural network factory available for SupervisedFeedForwardNeuralNetworks");
		}		
	}

	@Override
	public DirectedComponentFactory getDirectedComponentFactory() {
		return directedComponentFactoryReference;
	}

	@Override
	public ConvolutionalFeedForwardLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession> withConvolutionalLayer(
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
	public FeedForward3DLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, AveragePoolingFeedForwardLayerPropertiesBuilder<SupervisedFeedForwardNeuralNetwork3DBuilderSession>> withAveragePoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultAveragePoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}
	
	@Override
	public FeedForward3DLayerBuilderSession<SupervisedFeedForwardNeuralNetwork3DBuilderSession, Axons3DConfigBuilder, MaxPoolingFeedForwardLayerPropertiesBuilder<SupervisedFeedForwardNeuralNetwork3DBuilderSession>> withMaxPoolingLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultMaxPoolingFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}

	@Override
	public FullyConnectedFeedForwardLayerBuilderSession<SupervisedFeedForwardNeuralNetworkBuilderSession> withFullyConnectedLayer(
			String layerName) {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return new DefaultFullyConnectedFeedForwardLayerBuilderSession<>(layerName, directedLayerFactory, () -> this, getComponents()::add);
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkBuilderSession withComponentGraphDefinition(
			Component3DtoNon3DGraphDefinition definition) {
		definition.createComponentGraph(this, directedComponentFactory);
		return this;
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork3DBuilderSession withComponentGraphDefinition(
			Component3Dto3DGraphDefinition definition) {
		definition.createComponentGraph(this, directedComponentFactory);
		return this;
	}
	
	
	private Neurons3D getCurrentNeurons() {
		if (getComponents().isEmpty())  {
			return this.initialNeurons;
		} else {
			DefaultChainableDirectedComponent<?, ?> finalComponent = getComponents().get(getComponents().size() - 1);
			Neurons neurons = finalComponent.getOutputNeurons();
			if (neurons instanceof Neurons3D) {
				return (Neurons3D)neurons;
			} else {
				throw new IllegalStateException("Current neurons of 3D graph are expected to be instances of Neurons3D");
			}
		}
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork3DGraphBuilderSession with3DComponentGraph() {
		return new DefaultSupervisedFeedForwardNeuralNetwork3DGraphBuilderSession(this, getBuilderState(), directedComponentsContext);
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkGraphBuilderSession withComponentGraph() {
		return new DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession(this, getBuilderState().getNon3DBuilderState(), directedComponentsContext);
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>, L extends FeedForwardLayer<A, L>> SupervisedFeedForwardNeuralNetworkBuilderSession withLayer(
			L layer) {
		getComponents().add(layer);
		return new DefaultSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactoryReference, directedLayerFactory, neuralNetworkFactory, networkName, directedComponentsContext, getComponents(), getCurrentNeurons());
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>, L extends FeedForwardLayer<A, L>> SupervisedFeedForwardNeuralNetwork3DBuilderSession with3DLayer(
			L layer) {
		getComponents().add(layer);
		return new DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession(directedComponentFactoryReference, directedLayerFactory, neuralNetworkFactory, networkName, directedComponentsContext, getComponents(), getCurrentNeurons());
	}


}
