package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;

public abstract class UncompletedAxonsBuilderImpl<N extends Neurons, C> implements UncompletedAxonsBuilder<N, C> {

	protected Supplier<C> previousBuilderSupplier;
	protected N leftNeurons;
	protected DirectedComponentsContext directedComponentsContext;
	protected Consumer<AxonsContext> axonsContextConfigurer;
	
	public UncompletedAxonsBuilderImpl(Supplier<C> previousBuilderSupplier, N leftNeurons) {
		this.previousBuilderSupplier = previousBuilderSupplier;
		this.leftNeurons = leftNeurons;
	}
	
	public N getLeftNeurons() {
		return leftNeurons;
	}

	public DirectedComponentsContext getDirectedComponentsContext() {
		return directedComponentsContext;
	}

	public Consumer<AxonsContext> getAxonsContextConfigurer() {
		return axonsContextConfigurer;
	}
	
	
	
}
