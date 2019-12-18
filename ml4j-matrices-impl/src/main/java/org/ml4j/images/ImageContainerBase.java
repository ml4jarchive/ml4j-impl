package org.ml4j.images;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

public abstract class ImageContainerBase<I extends ImageContainer<I>> implements ImageContainer<I> {

	protected int width;
	protected int height;
	protected int paddingHeight = 0;
	protected int paddingWidth = 0;
	protected int examples;

	public ImageContainerBase(int width, int height, int paddingHeight, int paddingWidth, int examples) {
		this.width = width;
		this.height = height;
		this.paddingHeight = paddingHeight;
		this.paddingWidth = paddingWidth;
		this.examples = examples;
	}

	public abstract void populateData(float[] data, int startIndex);

	public abstract float[] getData();

	public abstract void close();

	public abstract void applyValueModifier(FloatPredicate condition, FloatModifier modifier);

	public abstract void applyValueModifier(FloatModifier modifier);

	public void setPaddingHeight(int paddingHeight) {
		this.paddingHeight = paddingHeight;
	}

	public void setPaddingWidth(int paddingWidth) {
		this.paddingWidth = paddingWidth;
	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

	public int getPaddingHeight() {
		return paddingHeight;
	}

	public int getPaddingWidth() {
		return paddingWidth;
	}

	public abstract void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2);

	public abstract int getDataLength();

	public abstract int getChannels();

	public abstract int getSubImageDataLength(int height, int width);

	public abstract void populateIm2col(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract void populateIm2col2(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract I dup();

	public abstract I softDup();

	public Matrix im2col(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		float[] data = new float[getChannels() * filterWidth * filterHeight * windowWidth * windowHeight * examples];
		populateIm2col(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
		return matrixFactory.createMatrixFromRowsByRowsArray(getChannels() * filterWidth * filterHeight,
				windowWidth * windowHeight * examples, data);
	}

	public Matrix im2col2(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		float[] data = new float[getChannels() * filterWidth * filterHeight * windowWidth * windowHeight * examples];
		populateIm2col2(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
		return matrixFactory.createMatrixFromRowsByRowsArray(filterWidth * filterHeight,
				windowWidth * windowHeight * examples * getChannels(), data);
	}
}
