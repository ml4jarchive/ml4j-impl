package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.GenericDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainImpl<A extends ChainableDirectedComponentActivation<NeuronsActivation>> extends GenericDirectedComponentChainImpl<NeuronsActivation, A> implements DefaultDirectedComponentChain<A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainImpl(List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends A, ?>> components) {
		super(components);
	}

	//@Override
	//protected DefaultDirectedComponentChain createDirectedComponentChain(
	//		List<ChainableDirectedComponent<NeuronsActivation, ?, ?, ?>> components) {
	//	return new DefaultGenericDirectedComponentChainImpl(components);
	//}

}
