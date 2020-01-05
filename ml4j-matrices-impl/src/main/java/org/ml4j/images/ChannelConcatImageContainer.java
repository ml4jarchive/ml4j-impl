package org.ml4j.images;

import java.util.List;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;

public abstract class ChannelConcatImageContainer<I extends ImageContainer<I>> extends ImageContainerBase<I> {

	protected List<I> channelConcatImages;

	public ChannelConcatImageContainer(List<I> channelConcatImages, int height, int width, int paddingHeight,
			int paddingWidth, int examples) {
		super(height, width, paddingHeight, paddingWidth, examples);
		this.channelConcatImages = channelConcatImages;
		for (I im : channelConcatImages) {
			if (im.getHeight() != height) {
				throw new IllegalArgumentException();
			}
			if (im.getWidth() != width) {
				throw new IllegalArgumentException();
			}
			if (im.getPaddingHeight() != paddingHeight) {
				throw new IllegalArgumentException();
			}
			if (im.getPaddingWidth() != paddingWidth) {
				throw new IllegalArgumentException();
			}
		}
	}

	@Override
	public void setPaddingHeight(int paddingHeight) {
		super.setPaddingHeight(paddingHeight);
		channelConcatImages.stream().forEach(i -> i.setPaddingHeight(paddingHeight));
	}

	@Override
	public void setPaddingWidth(int paddingWidth) {
		super.setPaddingWidth(paddingWidth);
		channelConcatImages.stream().forEach(i -> i.setPaddingWidth(paddingWidth));
	}

	public void populateData(float[] data, int startIndex) {
		int startIndex2 = startIndex;
		for (I subImage : channelConcatImages) {
			subImage.populateData(data, startIndex2);
			startIndex2 = startIndex2 + subImage.getDataLength();
		}
	}

	public void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startIndex2 = startIndex;
		for (I subImage : channelConcatImages) {
			subImage.populateDataSubImage(data, startIndex2, startHeight, startWidth, height, width, strideHeight,
					strideWidth, forIm2col2);
			startIndex2 = startIndex2 + subImage.getSubImageDataLength(height, width);
		}
	}
	
	public void populateDataSubImageReverse(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startIndex2 = startIndex;
		for (I subImage : channelConcatImages) {
			subImage.populateDataSubImageReverse(data, startIndex2, startHeight, startWidth, height, width, strideHeight,
					strideWidth, forIm2col2);
			startIndex2 = startIndex2 + subImage.getSubImageDataLength(height, width);
		}
	}

	public int getDataLength() {
		int length = 0;
		for (I subImage : channelConcatImages) {
			length = length + subImage.getDataLength();
		}
		return length;
	}

	@Override
	public int getSubImageDataLength(int height, int width) {
		return height * width * getChannels() * examples;
	}

	@Override
	public int getChannels() {
		int channels = 0;
		for (I subImage : channelConcatImages) {
			channels = channels + subImage.getChannels();
		}
		return channels;
	}

	@Override
	public void populateIm2colConvExport(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (I subImage : channelConcatImages) {
			subImage.populateIm2colConvExport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex
					+ windowWidth * windowHeight * examples * filterHeight * filterWidth * subImage.getChannels();
		}
	}

	public void populateIm2colPoolExport(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (I subImage : channelConcatImages) {
			subImage.populateIm2colPoolExport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex + examples * subImage.getChannels() * windowWidth * windowHeight;
		}
	}
	
	public void populateIm2colConvImport(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (I subImage : channelConcatImages) {
			subImage.populateIm2colConvImport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex
					+ windowWidth * windowHeight * examples * filterHeight * filterWidth * subImage.getChannels();
		}
	}

	public void populateIm2colPoolImport(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight,
			int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1) / strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1) / strideHeight;
		for (I subImage : channelConcatImages) {
			subImage.populateIm2colPoolImport(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex + examples * subImage.getChannels() * windowWidth * windowHeight;
		}
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		channelConcatImages.forEach(image -> image.applyValueModifier(condition, modifier));
	}

	@Override
	public void applyValueModifier(FloatModifier modifier) {
		channelConcatImages.forEach(image -> image.applyValueModifier(modifier));
	}

	@Override
	public void close() {
		channelConcatImages.forEach(i -> i.close());
	}

	@Override
	public float[] getData() {
		float[] data = new float[getDataLength()];
		populateData(data, 0);
		return data;
	}
}
