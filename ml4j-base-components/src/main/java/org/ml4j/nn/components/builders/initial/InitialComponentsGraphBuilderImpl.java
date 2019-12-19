package org.ml4j.nn.components.builders.initial;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.ComponentsGraphBuilderImpl;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderStateImpl;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;

public class InitialComponentsGraphBuilderImpl extends ComponentsGraphBuilderImpl<InitialComponentsGraphBuilder> implements InitialComponentsGraphBuilder {

	public InitialComponentsGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState, List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
	}
	
	public InitialComponentsGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, Neurons initialNeurons) {
		super(directedComponentFactory, new BaseGraphBuilderStateImpl(initialNeurons), new ArrayList<>());
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		return this;
	}
}
