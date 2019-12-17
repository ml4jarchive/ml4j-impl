package org.ml4j.nn.neurons;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.images.SingleChannelImages;

public class ImageNeuronsActivationImpl extends NeuronsActivationImpl implements ImageNeuronsActivation {

	private Neurons3D neurons;
	private Images images;
	private boolean immutable;

	public ImageNeuronsActivationImpl(Matrix activations, Neurons3D neurons,
			NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
		super(null, featureOrientation, immutable);
		this.neurons = neurons;
		if (neurons.getDepth() == 1) {
			images = new SingleChannelImages(activations.getRowByRowArray(), 0, neurons.getHeight(),
					neurons.getWidth(), 0, 0, activations.getColumns());
		} else {
			images = new MultiChannelImages(activations.getRowByRowArray(), neurons.getDepth(),
					neurons.getHeight(), neurons.getWidth(), 0, 0, activations.getColumns());
		}
	}

	@Override
	public void setImmutable(boolean immutable) {
		this.immutable = immutable;
	}

	@Override
	public boolean isImmutable() {
		return immutable;
	}

	@Override
	public int getExampleCount() {
		return images.getExamples();
	}

	@Override
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
		return this;
	}

	@Override
	public void close() {
		images.close();
	}

	@Override
	public Matrix getActivations(MatrixFactory matrixFactory) {
		Matrix activations = matrixFactory.createMatrixFromRowsByRowsArray(getRows(), getColumns(), images.getData());
		activations.setImmutable(immutable);
		return activations;
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		images.applyValueModifier(condition, modifier);
	}

	@Override
	public void applyValueModifier(FloatModifier modifier) {
		images.applyValueModifier(modifier);
	}

	public ImageNeuronsActivationImpl(Neurons3D neurons, Images image,
			NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
		super(null, featureOrientation, immutable);
		this.neurons = neurons;
		this.images = image;
	}

	@Override
	public NeuronsActivation dup() {
		return new ImageNeuronsActivationImpl(neurons, images, this.getFeatureOrientation(), immutable);
	}

	@Override
	public int getRows() {
		return neurons.getNeuronCountExcludingBias();
	}

	@Override
	public int getColumns() {
		return images.getExamples();
	}

	@Override
	public int getFeatureCount() {
		return getRows();
	}

	@Override
	public Neurons3D getNeurons() {
		return neurons;
	}

	public Images getImages() {
		return images;
	}

	public Matrix im2Col(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int paddingHeight, int paddingWidth) {
		Images imageWithPadding = images.softDup();
		imageWithPadding.setPaddingHeight(paddingHeight);
		imageWithPadding.setPaddingWidth(paddingWidth);
		return imageWithPadding.im2col(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
	}

	public Matrix im2Col2(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int paddingHeight, int paddingWidth) {
		Images imageWithPadding = images.softDup();
		imageWithPadding.setPaddingHeight(paddingHeight);
		imageWithPadding.setPaddingWidth(paddingWidth);
		return imageWithPadding.im2col2(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
	}

	@Override
	public NeuronsActivation transpose() {
		throw new UnsupportedOperationException();
	}
}
