package org.ml4j.nn.sessions;

import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.builders.initial.InitialComponentsGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

public class DefaultSupervisedFeedForwardNeuralNetworkBuilderSession extends InitialComponentsGraphBuilderImpl<DefaultChainableDirectedComponent<?, ?>>
		implements SupervisedFeedForwardNeuralNetworkBuilderSession{

	private String networkName;
	
	private DirectedLayerFactory directedLayerFactory;
	private SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory;
	private DirectedComponentFactory directedComponentFactoryReference;
	private Neurons initialNeurons;
	
	public DefaultSupervisedFeedForwardNeuralNetworkBuilderSession(
			DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory, String networkName,
			DirectedComponentsContext directedComponentsContext, List<DefaultChainableDirectedComponent<?, ?>> components, Neurons currentNeurons) {
		super(directedComponentFactory, directedComponentsContext, currentNeurons);
		this.directedComponentFactoryReference = directedComponentFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.neuralNetworkFactory = neuralNetworkFactory;
		this.networkName = networkName;
		this.initialNeurons = currentNeurons;
		this.getComponents().addAll(components);
	}

	public List<DefaultChainableDirectedComponent<?, ?>> getLayers() {
		return getComponents();
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
	public DirectedLayerFactory getDirectedLayerFactory() {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been set on the session");
		}
		return directedLayerFactory;
	}

	private Neurons getCurrentNeurons() {
		if (getComponents().isEmpty())  {
			return this.initialNeurons;
		} else {
			return getComponents().get(getComponents().size() - 1).getOutputNeurons();
		}
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
	public SupervisedFeedForwardNeuralNetworkGraphBuilderSession withComponentGraph() {
		return new DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession(this, getBuilderState() , directedComponentsContext);
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>, L extends FeedForwardLayer<A, L>> SupervisedFeedForwardNeuralNetworkBuilderSession withLayer(
			L layer) {
		getComponents().add(layer);
		return new DefaultSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactoryReference, directedLayerFactory, neuralNetworkFactory, networkName, directedComponentsContext, getComponents(), getCurrentNeurons());
	}
}
