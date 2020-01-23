package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBaseParent;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedLayerChainImpl<L extends DirectedLayer<?, ?>>
		extends DefaultDirectedComponentChainBaseParent<L, DirectedLayerActivation, DirectedLayerChainActivation>
		implements DirectedLayerChain<L> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedLayerChainImpl(List<L> components) {
		super(components);
	}

	@SuppressWarnings("unchecked")
	@Override
	public DirectedLayerChain<L> dup() {
		return new DirectedLayerChainImpl<L>(
				(List<L>) this.sequentialComponents.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

	@Override
	public NeuralComponentType<DirectedLayerChain<L>> getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentType.getBaseType(NeuralComponentBaseType.LAYER_CHAIN),
				getClass().getName());
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return sequentialComponents.stream().flatMap(c -> c.decompose().stream()).collect(Collectors.toList());
	}

	@Override
	public DirectedLayerChainActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		// LOGGER.debug("Forward propagating through DirectedLayerChainImpl");
		NeuronsActivation inFlightActivation = neuronsActivation;
		List<DirectedLayerActivation> activations = new ArrayList<>();
		for (L component : sequentialComponents) {
			DirectedLayerActivation activation = forwardPropagate(inFlightActivation, component, context);
			activations.add(activation);
			inFlightActivation = activation.getOutput();
		}

		return new DirectedLayerChainActivationImpl(activations);
	}

}
