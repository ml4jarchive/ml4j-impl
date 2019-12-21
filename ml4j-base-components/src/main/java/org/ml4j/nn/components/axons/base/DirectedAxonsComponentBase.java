package org.ml4j.nn.components.axons.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsContextImpl;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class DirectedAxonsComponentBase<L extends Neurons, R extends Neurons, A extends Axons<? extends L, ? extends R,  ?>> implements DirectedAxonsComponent<L, R> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedAxonsComponentBase.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	protected A axons;
	
	public DirectedAxonsComponentBase(A axons) {
		this.axons = axons;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public AxonsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return new AxonsContextImpl(context.getMatrixFactory(), context.isTrainingContext(), false);
	}

	@Override
	public Axons<? extends L, ? extends R, ?> getAxons() {
		return axons;
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.AXONS;
	}

	@Override
	public Neurons getInputNeurons() {
		return axons.getLeftNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return axons.getRightNeurons();
	}
	
	

}
