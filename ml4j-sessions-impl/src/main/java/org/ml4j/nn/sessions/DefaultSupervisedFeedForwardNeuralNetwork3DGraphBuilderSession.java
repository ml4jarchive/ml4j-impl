package org.ml4j.nn.sessions;

import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.Components3DGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderStateImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;

public class DefaultSupervisedFeedForwardNeuralNetwork3DGraphBuilderSession
	extends Components3DGraphBuilderImpl<SupervisedFeedForwardNeuralNetwork3DGraphBuilderSession, SupervisedFeedForwardNeuralNetworkGraphBuilderSession, DefaultChainableDirectedComponent<?, ?>>
		implements SupervisedFeedForwardNeuralNetwork3DGraphBuilderSession {

	private DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession previousBuilderSession;
	
	public DefaultSupervisedFeedForwardNeuralNetwork3DGraphBuilderSession(
			DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession previousBuilderSession, 
			Base3DGraphBuilderState builderState) {
		super(previousBuilderSession.getDirectedComponentFactory(), builderState, 
				previousBuilderSession.getComponents());
		this.previousBuilderSession = previousBuilderSession;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork build() {
		//addAxonsIfApplicable();
		return previousBuilderSession.getNeuralNetworkFactory()
				.createSupervisedFeedForwardNeuralNetwork(previousBuilderSession.getNetworkName(), this.getComponents());
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork3DBuilderSession endComponentGraph() {
		//addAxonsIfApplicable();
		return previousBuilderSession;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork3DGraphBuilderSession get3DBuilder() {
		return this;
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkGraphBuilderSession getBuilder() {
		//addAxonsIfApplicable();
		return new DefaultSupervisedFeedForwardNeuralNetworkGraphBuilderSession(this, 
				previousBuilderSession.getDirectedComponentFactory(), new BaseGraphBuilderStateImpl(this.getBuilderState().getComponentsGraphNeurons().getCurrentNeurons()));
	}


}
