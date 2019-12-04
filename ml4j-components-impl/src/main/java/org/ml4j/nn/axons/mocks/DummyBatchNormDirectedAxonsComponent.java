package org.ml4j.nn.axons.mocks;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.neurons.Neurons;

public class DummyBatchNormDirectedAxonsComponent<L extends Neurons, R extends Neurons> extends DummyDirectedAxonsComponent<L, R>
		implements BatchNormDirectedAxonsComponent<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DummyBatchNormDirectedAxonsComponent(MatrixFactory matrixFactory, Axons<? extends L, ? extends R, ?> axons) {
		super(matrixFactory, axons);
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		return null;
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		return null;
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix meansColumnVector) {
		// no-op
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix variancesColumnVector) {
		// no-op		
	}

	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		return 0;
	}

}
