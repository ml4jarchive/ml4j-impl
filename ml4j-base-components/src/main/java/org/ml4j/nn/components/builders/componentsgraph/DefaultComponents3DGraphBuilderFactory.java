package org.ml4j.nn.components.builders.componentsgraph;

import org.ml4j.nn.components.builders.initial.InitialComponents3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons3D;

public class DefaultComponents3DGraphBuilderFactory implements Components3DGraphBuilderFactory {

	private DirectedComponentFactory directedComponentFactory;
	
	public DefaultComponents3DGraphBuilderFactory(DirectedComponentFactory directedComponentFactory) {
		this.directedComponentFactory = directedComponentFactory;
	}
	
	@Override
	public InitialComponents3DGraphBuilder createInitialComponents3DGraphBuilder(Neurons3D initialNeurons) {
		return new InitialComponents3DGraphBuilderImpl(directedComponentFactory, initialNeurons);
	}


}
