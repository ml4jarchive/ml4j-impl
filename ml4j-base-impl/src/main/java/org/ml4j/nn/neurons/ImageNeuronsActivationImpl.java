/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.neurons;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.images.SingleChannelImages;
/**
 * Encapsulates the activation activities representing an Image of a set of Neurons3D neurons.
 * 
 * @author Michael Lavelle
 *
 */
public class ImageNeuronsActivationImpl extends NeuronsActivationImpl implements ImageNeuronsActivation {

	private Neurons3D neurons;
	private Images images;
	private boolean immutable;
	private Integer exampleCount;

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
	
	public ImageNeuronsActivationImpl(Neurons3D neurons,
			Images images, boolean immutable) {
		super(null, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, immutable);
		this.neurons = neurons;
		this.images = images;
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
		return exampleCount == null ? images.getExamples() : exampleCount;
	}

	@Override
	public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
		return this;
	}

	@Override
	public void close() {
		if (images != null && !images.isClosed()) {
			exampleCount = images.getExamples();
			images.close();
			images = null;
		}
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
		return new ImageNeuronsActivationImpl(neurons, images.dup(), this.getFeatureOrientation(), immutable);
	}

	@Override
	public int getRows() {
		return neurons.getNeuronCountExcludingBias();
	}

	@Override
	public int getColumns() {
		return getExampleCount();
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

	@Override
	public Matrix im2ColConv(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int paddingHeight, int paddingWidth) {
		Images imageWithPadding = images.softDup();
		imageWithPadding.setPaddingHeight(paddingHeight);
		imageWithPadding.setPaddingWidth(paddingWidth);
		return imageWithPadding.im2colConvExport(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
	}

	@Override
	public Matrix im2ColPool(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int paddingHeight, int paddingWidth) {
		Images imageWithPadding = images.softDup();
		imageWithPadding.setPaddingHeight(paddingHeight);
		imageWithPadding.setPaddingWidth(paddingWidth);
		return imageWithPadding.im2colPoolExport(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
	}

	@Override
	public NeuronsActivation transpose() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void reshape(int featureCount, int exampleCount) {
		throw new UnsupportedOperationException();
	}
	
	
}
