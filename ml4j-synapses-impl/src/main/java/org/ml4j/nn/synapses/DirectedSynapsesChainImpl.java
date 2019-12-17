package org.ml4j.nn.synapses;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.components.defaults.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesChainImpl<S extends DirectedSynapses<?, ?>> extends DirectedComponentChainBaseImpl<NeuronsActivation, S, DirectedSynapsesActivation, DirectedSynapsesChainActivation> implements DirectedSynapsesChain<S> {

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return super.decompose().stream().map(c -> adaptComponent(c)).collect(Collectors.toList());
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedSynapsesChainImpl(
			List<S> components) {
		super(components);
	}

	@Override
	protected DirectedSynapsesChainActivation createChainActivation(
			List<DirectedSynapsesActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DirectedSynapsesChainActivationImpl(componentActivations, inFlightInput);
	}

	@SuppressWarnings("unchecked")
	@Override
	public DirectedSynapsesChain<S> dup() {
		return new DirectedSynapsesChainImpl<S>((List<S>) components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

	protected <C> DefaultChainableDirectedComponentAdapter<?> adaptComponent(ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> c) {
		return new DefaultChainableDirectedComponentAdapter<>(c);
	}

}
