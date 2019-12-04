package org.ml4j.nn.components.builders.componentsgraph;

import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;

public class DefaultComponents3DGraphBuilderFactory implements Components3DGraphBuilderFactory {

	private DirectedAxonsComponentFactory directedAxonsComponentFactory;
	
	public DefaultComponents3DGraphBuilderFactory(DirectedAxonsComponentFactory directedAxonsComponentFactory) {
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
	}
	
	@Override
	public InitialComponents3DGraphBuilder createInitialComponents3DGraphBuilder(Neurons3D initialNeurons) {
		return new InitialComponents3DGraphBuilderImpl(directedAxonsComponentFactory, initialNeurons);
	}


}
