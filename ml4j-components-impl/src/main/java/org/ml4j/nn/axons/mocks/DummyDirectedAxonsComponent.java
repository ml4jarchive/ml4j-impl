package org.ml4j.nn.axons.mocks;

import java.util.Arrays;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsContextImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedAxonsComponent<L extends Neurons, R extends Neurons> implements DirectedAxonsComponent<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private MatrixFactory matrixFactory;
	private Axons<? extends L, ? extends R, ?> axons;
	
	public DummyDirectedAxonsComponent(MatrixFactory matrixFactory, Axons<? extends L, ? extends R, ?> axons)  {
		this.matrixFactory = matrixFactory;
		this.axons = axons;
	}

	@Override
	public AxonsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return new AxonsContextImpl(matrixFactory, false);
	}

	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation inputActivation, AxonsContext context) {
		AxonsActivation axonsActivation = axons.pushLeftToRight(inputActivation, null, context);
		return new DummyDirectedAxonsComponentActivation(this, axonsActivation.getPostDropoutInput(), axonsActivation.getOutput());
	}

	@Override
	public Axons<? extends L, ? extends R, ?> getAxons() {
		return axons;
	}

}
