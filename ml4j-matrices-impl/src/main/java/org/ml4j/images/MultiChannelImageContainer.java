package org.ml4j.images;

import java.util.List;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;

public abstract class MultiChannelImageContainer<I extends ImageContainer<I>> extends ImageContainerBase<I> {

	protected float[] data;
	protected int channels;

	protected boolean closed;

	@Override
	public boolean isClosed() {
		return closed;
	}

	public MultiChannelImageContainer(float[] data, int channels, int height, int width, int paddingHeight,
			int paddingWidth, int examples) {
		super(height, width, paddingHeight, paddingWidth, examples);
		this.data = data;
		this.channels = channels;
		if (data == null) {
			throw new IllegalArgumentException();
		}
	}

	public void populateData(float[] data, int startIndex) {
		if (paddingHeight == 0 && paddingWidth == 0) {
			System.arraycopy(this.data, 0, data, startIndex, getDataLength());
		} else {
			populateDataSubImage(data, startIndex, 0, 0, height, width, 1, 1, false);
		}
	}

	@Override
	public float[] getData() {
		if (paddingHeight == 0 && paddingWidth == 0 && data.length == getDataLength()) {
			return data;
		} else {
			float[] populatedData = new float[getDataLength()];
			populateData(populatedData, 0);
			return populatedData;
		}
	}

	public int getDataLength() {
		return height * width * examples * channels;
	}

	protected abstract List<I> getChannelConcatImages();

	public void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startIndex2 = startIndex;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateDataSubImage(data, startIndex2, startHeight, startWidth, height, width, strideHeight,
					strideWidth, forIm2col2);
			startIndex2 = startIndex2 + subImage.getSubImageDataLength(height, width);
		}

	}

	public void populateDataSubImageReverse(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startIndex2 = startIndex;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateDataSubImageReverse(data, startIndex2, startHeight, startWidth, height, width,
					strideHeight, strideWidth, forIm2col2);
			startIndex2 = startIndex2 + subImage.getSubImageDataLength(height, width);
		}

	}

	@Override
	public int getSubImageDataLength(int height, int width) {
		return height * width * examples * channels;
	}

	@Override
	public int getChannels() {
		return channels;
	}

	@Override
	public void populateIm2colConvExport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateIm2colConvExport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth,
					channels);
			startIndex = startIndex
					+ windowWidth * windowHeight * examples * filterHeight * filterWidth * subImage.getChannels();
		}
	}

	@Override
	public void populateIm2colConvImport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateIm2colConvImport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth,
					channels);
			startIndex = startIndex
					+ windowWidth * windowHeight * examples * filterHeight * filterWidth * subImage.getChannels();
		}
	}

	@Override
	public void populateIm2colPoolExport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateIm2colPoolExport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth,
					channels);
			startIndex = startIndex + examples * subImage.getChannels() * windowWidth * windowHeight;
		}
	}

	@Override
	public void populateIm2colPoolImport(float[] data, int startIndex, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (ImageContainer<?> subImage : getChannelConcatImages()) {
			subImage.populateIm2colPoolImport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth,
					channels);
			startIndex = startIndex + examples * subImage.getChannels() * windowWidth * windowHeight;
		}
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		for (int i = 0; i < getDataLength(); i++) {
			float v = data[i];
			if (condition.test(v)) {
				data[i] = modifier.acceptAndModify(data[i]);
			}
		}
	}

	@Override
	public void applyValueModifier(FloatModifier modifier) {
		for (int i = 0; i < getDataLength(); i++) {
			data[i] = modifier.acceptAndModify(data[i]);
		}
	}

	@Override
	public void close() {
		this.closed = true;
		this.data = null;
	}
}
