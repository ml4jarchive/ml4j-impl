package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;

public class UncompletedFullyConnectedAxonsBuilderImpl<C extends AxonsBuilder> extends UncompletedAxonsBuilderImpl<Neurons, C> implements UncompletedFullyConnectedAxonsBuilder<C> {

	public UncompletedFullyConnectedAxonsBuilderImpl(Supplier<C> previousBuilderSupplier, Neurons leftNeurons) {
		super(previousBuilderSupplier, leftNeurons);
	}

	@Override
	public C withConnectionToNeurons(Neurons neurons) {
		previousBuilderSupplier.get().getComponentsGraphNeurons().setRightNeurons(neurons);
		return previousBuilderSupplier.get();
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withConnectionWeights(Matrix connectionWeights) {
		previousBuilderSupplier.get().getBuilderState().setConnectionWeights(connectionWeights);
		return this;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withBiasUnit() {
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withBiases(Matrix biases) {
		previousBuilderSupplier.get().getBuilderState().setBiases(biases);
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}
	
	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withAxonsContext(
			DirectedComponentsContext directedComponentsContext, Consumer<AxonsContext> axonsContextConfigurer) {
		this.directedComponentsContext = directedComponentsContext;
		this.axonsContextConfigurer = axonsContextConfigurer;
		return this;
	}


}
