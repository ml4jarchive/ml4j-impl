package org.ml4j.nn.components.builders.initial;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderStateImpl;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class InitialComponentsGraphBuilderImpl extends ComponentsGraphBuilderImpl<InitialComponentsGraphBuilder> implements InitialComponentsGraphBuilder {

	public InitialComponentsGraphBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, BaseGraphBuilderState builderState, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
	}
	
	public InitialComponentsGraphBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Neurons initialNeurons) {
		super(directedAxonsComponentFactory, new BaseGraphBuilderStateImpl(initialNeurons), new ArrayList<>());
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		return this;
	}
}
