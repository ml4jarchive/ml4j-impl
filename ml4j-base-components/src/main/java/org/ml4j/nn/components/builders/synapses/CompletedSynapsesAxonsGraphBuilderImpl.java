package org.ml4j.nn.components.builders.synapses;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsGraphSkipConnectionBuilder;
import org.ml4j.nn.components.builders.axonsgraph.AxonsSubGraphBuilder;
import org.ml4j.nn.components.builders.base.BaseGraphBuilderImpl;
import org.ml4j.nn.components.builders.common.AxonsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.skipconnection.AxonsGraphSkipConnectionBuilderImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;

public class CompletedSynapsesAxonsGraphBuilderImpl<P extends AxonsBuilder> extends BaseGraphBuilderImpl<CompletedSynapsesAxonsGraphBuilder<P>> implements CompletedSynapsesAxonsGraphBuilder<P>, SynapsesEnder<P> {

	private Supplier<P> previousSupplier;
	
	public CompletedSynapsesAxonsGraphBuilderImpl(Supplier<P> previousSupplier, DirectedComponentFactory directedComponentFactory,
			BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
		this.previousSupplier = previousSupplier;
	}

	@Override
	public ParallelPathsBuilder<AxonsSubGraphBuilder<CompletedSynapsesAxonsGraphBuilder<P>>> withParallelPaths() {
		return new AxonsParallelPathsBuilderImpl<>(directedComponentFactory,() -> this);
	}
	
	@Override
	public AxonsGraphSkipConnectionBuilder<CompletedSynapsesAxonsGraphBuilder<P>> withSkipConnection() {
		return new AxonsGraphSkipConnectionBuilderImpl<>(this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
	}

	@Override
	public SynapsesEnder<P> withActivationFunction(
			DifferentiableActivationFunction activationFunction) {
		addActivationFunction(activationFunction);
		return this;
	}

	@Override
	public P endSynapses() {
		addAxonsIfApplicable();
		this.previousSupplier.get().addAxonsIfApplicable();
		this.previousSupplier.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
		this.previousSupplier.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
		this.previousSupplier.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
		// TODO ML Here we would add the synapses instead of the chain
		DefaultDirectedComponentChain
			chain = directedComponentFactory.createDirectedComponentChain(this.getComponents());
		previousSupplier.get().addComponent(chain);
		
		return previousSupplier.get();
	}

	@Override
	public CompletedSynapsesAxonsGraphBuilder<P> getBuilder() {
		return this;
	}
}
