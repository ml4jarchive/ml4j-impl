package org.ml4j.nn.components.axons.base;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.Neurons;

public class BatchNormDirectedAxonsComponentAdapter<N extends Neurons> extends DefaultChainableDirectedComponentAdapter<DirectedAxonsComponentActivation, AxonsContext> 
	implements BatchNormDirectedAxonsComponent<N, Axons<N, N, ?>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public BatchNormDirectedAxonsComponentAdapter(
			BatchNormDirectedAxonsComponent<N, ?> delegated) {
		super(delegated, BatchNormDirectedAxonsComponent.class.getSimpleName() + ":" + ((Axons<N, N, ?>)((BatchNormDirectedAxonsComponent<N, ?>)delegated).getAxons()).getClass().getSimpleName());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Axons<N, N, ?> getAxons() {
		return (Axons<N, N, ?>)((BatchNormDirectedAxonsComponent<N, ?>)delegated).getAxons();
	}

	@SuppressWarnings("unchecked")
	@Override
	public BatchNormDirectedAxonsComponentAdapter<N> dup() {
		return new BatchNormDirectedAxonsComponentAdapter<N>( (BatchNormDirectedAxonsComponent<N, ?>) delegated.dup());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getExponentiallyWeightedAverageInputFeatureMeans();
	}

	@SuppressWarnings("unchecked")
	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getExponentiallyWeightedAverageInputFeatureVariances();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix meansColumnVector) {
		((BatchNormDirectedAxonsComponent<N, ?>)delegated).setExponentiallyWeightedAverageInputFeatureMeans(meansColumnVector);
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix variancesColumnVector) {
		((BatchNormDirectedAxonsComponent<N, ?>)delegated).setExponentiallyWeightedAverageInputFeatureVariances(variancesColumnVector);		
	}

	@SuppressWarnings("unchecked")
	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		return ((BatchNormDirectedAxonsComponent<N, ?>)delegated).getBetaForExponentiallyWeightedAverages();
	}

}
