package org.ml4j.nn.sessions;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;

public class DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession extends
	ComponentsGraphBuilderImpl<SupervisedFeedForwardNeuralNetworkGraphBuilderSession, DefaultChainableDirectedComponent<?, ?>>
		implements SupervisedFeedForwardNeuralNetworkGraphBuilderSession {

	private SupervisedFeedForwardNeuralNetworkBuilderSession previousBuilderSession;
	
	public DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession(
			SupervisedFeedForwardNeuralNetworkBuilderSession previousBuilderSession,
			BaseGraphBuilderState builderState) {
		super(previousBuilderSession.getDirectedComponentFactory(), builderState, 
				previousBuilderSession.getComponents());
		this.previousBuilderSession = previousBuilderSession;		
	}

	public DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession(
			SupervisedFeedForwardNeuralNetwork3DGraphBuilderSession previousBuilderSession,
			DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState) {
		super(directedComponentFactory, builderState, 
				previousBuilderSession.getComponents());		
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetwork build() {
		if (previousBuilderSession == null) {
			throw new IllegalStateException("No component graph has been started");
		}
		//addAxonsIfApplicable();
		return previousBuilderSession.getNeuralNetworkFactory()
				.createSupervisedFeedForwardNeuralNetwork(previousBuilderSession.getNetworkName(), components);
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkBuilderSession endComponentGraph() {
		if (previousBuilderSession == null) {
			throw new IllegalStateException("No component graph has been started");
		}
		//addAxonsIfApplicable();
		return previousBuilderSession;
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkGraphBuilderSession getBuilder() {
		//addAxonsIfApplicable();
		return this;
	}

	
}
