package org.ml4j.images;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;

public abstract class SingleChannelImageContainer<I extends ImageContainer<I>> extends ImageContainerBase<I> {

	protected float[] data;
	protected int startIndex;

	public SingleChannelImageContainer(float[] data, int startIndex, int height, int width, int paddingHeight,
			int paddingWidth, int examples) {
		super(height, width, paddingHeight, paddingWidth, examples);
		this.data = data;
		this.startIndex = startIndex;
		this.examples = examples;
		this.width = width;
		this.height = height;
		this.paddingWidth = paddingWidth;
		this.paddingHeight = paddingHeight;
	}

	public void populateData(float[] data, int startIndex) {
		if (paddingHeight == 0 && paddingWidth == 0) {
			System.arraycopy(this.data, this.startIndex, data, startIndex, getDataLength());
		} else {
			populateDataSubImage(data, startIndex, 0, 0, height, width, 1, 1, false);
		}
	}

	@Override
	public float[] getData() {
		if (paddingHeight == 0 && paddingWidth == 0 && startIndex == 0 && data.length == getDataLength()) {
			return data;
		} else {
			float[] populatedData = new float[getDataLength()];
			populateData(populatedData, startIndex);
			return populatedData;
		}
	}

	public int getDataLength() {
		return height * width * examples;
	}

	@Override
	public void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startH = startHeight - paddingHeight;
		for (int sourceH = startH; sourceH < startH + this.height; sourceH += strideHeight) {
			int targetH = (sourceH - startH) / strideHeight;
			if (sourceH >= 0 && targetH >= 0 && sourceH < this.height && targetH < height) {
				if (strideWidth == 1) {
					int startW2 = Math.max(startWidth - paddingWidth, 0);
					int widthToCopy = Math.min(width - paddingWidth + (forIm2col2 ? 0 : startWidth),
							this.width - startW2);
					int startW = Math.max(paddingWidth - startWidth, 0);
					System.arraycopy(this.data, this.startIndex + sourceH * this.width * examples + startW2 * examples,
							data, startIndex + targetH * width * examples + startW * examples,
							examples * (widthToCopy));
				} else {
					int widthToCopy = 1;
					int startW2 = startWidth - paddingWidth;
					int startW = Math.max(paddingWidth - startWidth, 0);
					for (int w = startW2; w < this.width; w += strideWidth) {
						if (w >= 0) {
							System.arraycopy(this.data,
									this.startIndex + sourceH * this.width * examples + w * examples, data,
									startIndex + targetH * width * examples + startW * examples,
									examples * (widthToCopy));
							startW = startW + 1;
						}

					}
				}
			}
		}
	}

	@Override
	public int getSubImageDataLength(int height, int width) {
		return height * width * examples;
	}

	@Override
	public int getChannels() {
		return 1;
	}

	@Override
	public void populateIm2col(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (int h = 0; h < filterHeight; h++) {
			for (int w = 0; w < filterWidth; w++) {
				populateDataSubImage(data, startIndex, h, w, windowHeight, windowWidth, strideHeight, strideWidth,
						false);
				startIndex = startIndex + getSubImageDataLength(windowHeight, windowWidth);
			}
		}
	}

	@Override
	public void populateIm2col2(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (int h = 0; h < filterHeight; h++) {
			for (int w = 0; w < filterWidth; w++) {
				populateDataSubImage(data, startIndex, h, w, windowHeight, windowWidth, strideHeight, strideWidth,
						true);
				startIndex = startIndex + getSubImageDataLength(windowHeight, windowWidth) * channels;
			}
		}
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		for (int i = startIndex; i < startIndex + getDataLength(); i++) {
			if (condition.test(data[i])) {
				data[i] = modifier.acceptAndModify(data[i]);
			}
		}
	}

	@Override
	public void applyValueModifier(FloatModifier modifier) {
		for (int i = startIndex; i < startIndex + getDataLength(); i++) {
			data[i] = modifier.acceptAndModify(data[i]);
		}
	}

	@Override
	public void close() {
		this.data = null;
	}
}
