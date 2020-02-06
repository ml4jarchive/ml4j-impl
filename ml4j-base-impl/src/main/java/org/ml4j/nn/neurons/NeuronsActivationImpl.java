package org.ml4j.nn.neurons;

import org.ml4j.EditableMatrix;
import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.ImageFeaturesFormat;

public class NeuronsActivationImpl implements NeuronsActivation {

	protected Matrix activations;
	private boolean immutable;
	private NeuronsActivationFormat<?> format;
	private Neurons neurons;

	public NeuronsActivationImpl(Neurons neurons, Matrix activations,
			NeuronsActivationFormat<?> format, boolean immutable) {
		this.neurons = neurons;
		this.activations = activations;
		this.format = format;
		this.immutable = immutable;

		if (neurons == null) {
			throw new IllegalArgumentException();
		}
	}

	public NeuronsActivationImpl(Neurons neurons, Matrix activations,
			NeuronsActivationFormat<?> format) {
		this(neurons, activations, format, false);
	}

	@Override
	public void addInline(MatrixFactory matrixFactory, NeuronsActivation other) {
		EditableMatrix editableActivations = activations.asEditableMatrix();
		if (!other.getFormat().equals(format)) {
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
		if (!other.getFormat().equals(format)) {
			throw new IllegalArgumentException("Incompatible orientations");
		}
		if (format.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
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
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons, DimensionScope dimensionScope) {
		if (neurons.getNeuronCountIncludingBias() != neurons.getNeuronCountIncludingBias()) {
			throw new IllegalArgumentException();
		}
		// TODO scope
		if (format.getFeaturesFormat() instanceof ImageFeaturesFormat) {
			ImageNeuronsActivationFormat imageFeaturesFormat = new ImageNeuronsActivationFormat(getFeatureOrientation(), (ImageFeaturesFormat)format.getFeaturesFormat(), format.getExampleDimensions());
			return new ImageNeuronsActivationImpl(activations, neurons, imageFeaturesFormat,
					activations.isImmutable() || immutable);
		} else {
			if (ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT
					.isEquivalentFormat(format, dimensionScope)) {
				return new ImageNeuronsActivationImpl(activations, neurons, ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT,
						activations.isImmutable() || immutable);
				
			} else {
				throw new IllegalStateException("Format is not a compatible image format:" + format);
			}
		}
		
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
		return new NeuronsActivationImpl(neurons, activations.dup(), format, false);
	}

	@Override
	public Matrix getActivations(MatrixFactory matrixFactory) {
		if (activations == null || activations.isClosed()) {
			throw new IllegalStateException("Matrix has been closed");
		}
		return activations;
	}

	@Override
	public int getColumns() {
		return activations.getColumns();
	}

	@Override
	public int getExampleCount() {
		return format.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? getColumns()
				: getRows();
	}

	@Override
	public int getFeatureCount() {
		return format.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? getRows()
				: getColumns();
	}

	@Override
	public NeuronsActivationFeatureOrientation getFeatureOrientation() {
		return format.getFeatureOrientation();
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
		if (format.getFeatureOrientation() == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
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

	@Override
	public NeuronsActivationFormat<?> getFormat() {
		return format;
	}

}
