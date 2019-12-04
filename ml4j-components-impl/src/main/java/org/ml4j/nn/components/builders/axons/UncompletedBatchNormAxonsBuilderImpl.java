package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons3D;

public class UncompletedBatchNormAxonsBuilderImpl<C extends Axons3DBuilder> extends UncompletedAxonsBuilderImpl<Neurons3D, C> implements UncompletedBatchNormAxonsBuilder<C> {

	private Matrix gamma;
	private Matrix beta;
	private Matrix mean;
	private Matrix variance;
	
	public UncompletedBatchNormAxonsBuilderImpl(Supplier<C> previousBuilderSupplier, Neurons3D leftNeurons) {
		super(previousBuilderSupplier, leftNeurons);
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withGamma(Matrix gamma) {
		this.gamma = gamma;
		return this;
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withBeta(Matrix beta) {
		this.beta = beta;
		return this;
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withMean(Matrix mean) {
		this.mean = mean;
		return this;
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withVariance(Matrix variance) {
		this.variance = variance;
		return this;
	}

	@Override
	public Matrix getGamma() {
		return gamma;
	}

	@Override
	public Matrix getBeta() {
		return beta;
	}

	@Override
	public Matrix getMean() {
		return mean;
	}

	@Override
	public Matrix getVariance() {
		return variance;
	}

	@Override
	public C withConnectionToNeurons(Neurons3D neurons) {
		previousBuilderSupplier.get().getComponentsGraphNeurons().setRightNeurons(neurons);
		return previousBuilderSupplier.get();
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<C> withConnectionWeights(Matrix connectionWeights) {
		previousBuilderSupplier.get().getBuilderState().setConnectionWeights(connectionWeights);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<C> withBiasUnit() {
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<C> withBiases(Matrix biases) {
		previousBuilderSupplier.get().getBuilderState().setBiases(biases);
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withAxonsContext(
			DirectedComponentsContext directedComponentsContext, Consumer<AxonsContext> axonsContextConfigurer) {
		this.directedComponentsContext = directedComponentsContext;
		this.axonsContextConfigurer = axonsContextConfigurer;
		return this;
	}
}
