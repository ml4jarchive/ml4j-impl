package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.Matrix;

public class AxonWeightsAdjustmentImpl implements AxonWeightsAdjustment {

	private Matrix connectionWeights;
	private Matrix leftToRightBiases;
	private Matrix rightToLeftBiases;
	
	public AxonWeightsAdjustmentImpl(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
	}
	
	public AxonWeightsAdjustmentImpl(Matrix connectionWeights, Matrix leftToRightBiases) {
		this.connectionWeights = connectionWeights;
		this.leftToRightBiases = leftToRightBiases;
	}
	
	public AxonWeightsAdjustmentImpl(Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		this.connectionWeights = connectionWeights;
		this.leftToRightBiases = leftToRightBiases;
		this.rightToLeftBiases = rightToLeftBiases;
	}
	
	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public Optional<Matrix> getLeftToRightBiases() {
		return Optional.ofNullable(leftToRightBiases);
	}

	@Override
	public Optional<Matrix> getRightToLeftBiases() {
		return Optional.ofNullable(rightToLeftBiases);
	}
}
