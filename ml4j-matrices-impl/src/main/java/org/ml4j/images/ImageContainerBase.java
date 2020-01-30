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

	public abstract void populateIm2colConvExport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract void populateIm2colPoolExport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract void populateIm2colConvImport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract void populateIm2colPoolImport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels);

	public abstract I dup();

	public abstract I softDup();

	@Override
	public Matrix im2colConvExport(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		float[] data = new float[getChannels() * filterWidth * filterHeight * windowWidth * windowHeight * examples];
		populateIm2colConvExport(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
		return matrixFactory.createMatrixFromRowsByRowsArray(getChannels() * filterWidth * filterHeight,
				windowWidth * windowHeight * examples, data);
	}

	@Override
	public void im2colConvImport(MatrixFactory matrixFactory, Matrix matrix, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth) {
		float[] data = matrix.getRowByRowArray();
		populateIm2colConvImport(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
	}
	

	@Override
	public Matrix spaceToDepthExport(MatrixFactory matrixFactory, int blockHeight, int blockWidth) {
		float[] data = new float[getDataLength()];
		populateSpaceToDepthExport(data, 0, blockHeight, blockHeight);
		return matrixFactory.createMatrixFromRowsByRowsArray(getDataLength() / examples,
				examples, data);
	}

	@Override
	public void spaceToDepthImport(MatrixFactory matrixFactory, Matrix matrix, int heightFactor, int widthFactor) {
		float[] data = matrix.getRowByRowArray();
		populateSpaceToDepthImport(data, 0, heightFactor, widthFactor);
	}

	@Override
	public Matrix im2colPoolExport(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		float[] data = new float[getChannels() * filterWidth * filterHeight * windowWidth * windowHeight * examples];
		populateIm2colPoolExport(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
		return matrixFactory.createMatrixFromRowsByRowsArray(filterWidth * filterHeight,
				windowWidth * windowHeight * examples * getChannels(), data);
	}

	@Override
	public void im2colPoolImport(MatrixFactory matrixFactory, Matrix matrix, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth) {
		float[] data = matrix.getRowByRowArray();
		populateIm2colPoolImport(data, 0, filterHeight, filterWidth, strideHeight, strideWidth, getChannels());
	}
}
