package org.ml4j.nn.neurons;

import org.ml4j.EditableMatrix;
import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

public class NeuronsActivationImpl implements NeuronsActivation {
	
	protected Matrix activations;
	private boolean immutable;
	private NeuronsActivationFeatureOrientation featureOrientation;
	private Neurons neurons;
	
	public NeuronsActivationImpl( Neurons neurons, Matrix activations, NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
		this.neurons = neurons;
		this.activations = activations;
		this.featureOrientation = featureOrientation;
		this.immutable = immutable;
	
		if (neurons == null) {
			throw new IllegalArgumentException();
		}
		
		/*
		if (neurons.getNeuronCountExcludingBias() != (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? activations.getRows() : activations.getColumns())) {
			throw new IllegalArgumentException();
		}
		*/
	
	}
	
	public NeuronsActivationImpl( Neurons neurons, Matrix activations, NeuronsActivationFeatureOrientation featureOrientation) {
		this(neurons, activations, featureOrientation, false);
	}

	@Override
	public void addInline(MatrixFactory matrixFactory, NeuronsActivation other) {
		EditableMatrix editableActivations = activations.asEditableMatrix();
		if (other.getFeatureOrientation() != featureOrientation) {
			throw new IllegalArgumentException("Incompatible orientations");
		}
		if (other.getFeatureCount() != getFeatureCount()) {
			throw new IllegalArgumentException("Incompatible activations");
		}
		if (other.getExampleCount() != getExampleCount()) {
			throw new IllegalArgumentException("Incompatible activations");
		}
		editableActivations.addi(other.getActivations(matrixFactory));
	}
	

	@Override
	public void combineFeaturesInline(NeuronsActivation other, MatrixFactory matrixFactory) {
		if (other.getFeatureOrientation() != featureOrientation) {
			throw new IllegalArgumentException("Incompatible orientations");
		}
		if (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			activations = activations.appendVertically(other.getActivations(matrixFactory));
		} else {
			try (InterrimMatrix previousActivations = this.activations.asInterrimMatrix()) {
				this.activations = previousActivations.appendHorizontally(other.getActivations(matrixFactory));
			}
		}
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		EditableMatrix editableActivations = activations.asEditableMatrix();
		for (int i = 0; i < activations.getLength(); i++) {
			if (condition.test(activations.get(i))) {
				editableActivations.put(i, modifier.acceptAndModify(activations.get(i)));
			}
		}
	}

	@Override
	public void applyValueModifier(FloatModifier modifier) {
		EditableMatrix editableActivations = activations.asEditableMatrix();
		for (int i = 0; i < activations.getLength(); i++) {
			editableActivations.put(i, modifier.acceptAndModify(activations.get(i)));
		}
	}

	@Override
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
		if (neurons.getNeuronCountIncludingBias() != neurons.getNeuronCountIncludingBias()) {
			throw new IllegalArgumentException();
		}
		return new ImageNeuronsActivationImpl(activations, neurons, featureOrientation, activations.isImmutable() || immutable);		
	}

	@Override
	public void close() {
		if (activations != null && !activations.isClosed()) {
			activations.close();
			activations = null;
		}
	}


	@Override
	public NeuronsActivation dup() {
		return new NeuronsActivationImpl(neurons, activations.dup(), featureOrientation, false);
	}

	@Override
	public Matrix getActivations(MatrixFactory matrixFactory) {
		if (activations == null || activations.isClosed()) {
			throw new IllegalStateException();
		}
		return activations;
	}

	@Override
	public int getColumns() {
		return activations.getColumns();
	}

	@Override
	public int getExampleCount() {
		return featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? getColumns() : getRows();
	}

	@Override
	public int getFeatureCount() {
		return featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? getRows() : getColumns();
	}

	@Override
	public NeuronsActivationFeatureOrientation getFeatureOrientation() {
		return featureOrientation;
	}

	@Override
	public Neurons getNeurons() {
		return neurons;
	}

	@Override
	public int getRows() {
		return activations.getRows();
	}

	@Override
	public boolean isImmutable() {
		return immutable;
	}

	@Override
	public void reshape(int featureCount, int exampleCount) {
		if (this.immutable) {
			throw new IllegalStateException();
		}
		if (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			activations.asEditableMatrix().reshape(featureCount, exampleCount);		
		} else {
			activations.asEditableMatrix().reshape(exampleCount, featureCount);		
		}
	}

	@Override
	public void setImmutable(boolean immutable) {
		this.immutable = immutable;
		this.activations.setImmutable(immutable);
	}

}
