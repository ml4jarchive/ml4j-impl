package org.ml4j.images;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;

public class MultiChannelImage extends ImageBase {
		
	private float[] data;
	private int channels;
	private String closeStackTrace;
	
	public MultiChannelImage(float[] data, int channels, int height, int width, int paddingHeight, int paddingWidth, int examples) {
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
	
	private List<Image> getChannelConcatImages() {
		List<Image> channelConcatImages = new ArrayList<>();
		int sourceStartIndex = 0;
		for (int c = 0; c < channels; c++) {
			SingleChannelImage channelImage = new SingleChannelImage(data, sourceStartIndex, height, width, paddingHeight, paddingWidth, examples);
			channelConcatImages.add(channelImage);
			sourceStartIndex = sourceStartIndex + channelImage.getDataLength();
		}
		return channelConcatImages;
	}
	
	public void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height, int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startIndex2 = startIndex;
		 for (Image subImage : getChannelConcatImages()) {
			 subImage.populateDataSubImage(data, startIndex2, startHeight, startWidth, height, width, strideHeight, strideWidth, forIm2col2);
			 startIndex2 = startIndex2 + subImage.getSubImageDataLength(height, width);
		 }
	
	}

	/*
	@Override
	public void populateDataSubImage(float[] data, int startIndex, int startHeight, int startWidth, int height,
			int width, int strideHeight, int strideWidth, boolean forIm2col2) {
		int startH = startHeight - paddingHeight;
		//int startW = startWidth - paddingWidth;
		for (int sourceH = startH; sourceH < startH + this.height; sourceH+=strideHeight) {
			int targetH = (sourceH - startH)/strideHeight;
			if (sourceH >= 0 && targetH >= 0 && sourceH < this.height && targetH < height) {
				if (strideWidth == 1) {
					int startW2= Math.max(startWidth - paddingWidth, 0);
					int widthToCopy = Math.min(width - paddingWidth + (forIm2col2 ? 0 : startWidth), this.width - startW2);
					int startW = Math.max(paddingWidth - startWidth , 0);
					System.arraycopy(this.data, this.startIndex + sourceH * this.width * examples + startW2 * examples, data, startIndex + targetH * width * examples + startW * examples, examples * (widthToCopy));
				} else {
					int widthToCopy = 1;
					int startW2= startWidth - paddingWidth;
					int startW = Math.max(paddingWidth - startWidth, 0);
					for (int w = startW2; w < this.width; w+=strideWidth) { 
						if (w >= 0 ) {
						System.arraycopy(this.data, this.startIndex + sourceH * this.width * examples + w * examples, data, startIndex + targetH * width * examples + startW * examples, examples * (widthToCopy));
						startW = startW + 1;
						}

					}
				}
			}
		}
	}
	*/

	@Override
	public int getSubImageDataLength(int height, int width) {
		return height * width * examples * channels;
	}

	@Override
	public int getChannels() {
		return channels;
	}
	

	@Override
	public void populateIm2col(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1)/strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1)/strideHeight;
		for (Image subImage : getChannelConcatImages()) {
			subImage.populateIm2col(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex + windowWidth * windowHeight * examples * filterHeight * filterWidth * subImage.getChannels();
		}
	}
	
	public void populateIm2col2(float[] data, int startIndex, int filterHeight, int filterWidth, int strideHeight, int strideWidth, int channels) {
		int windowSpanWidth = width + 2 * paddingWidth - filterWidth + 1;
		int windowSpanHeight = height + 2 * paddingHeight - filterHeight + 1;
		int windowWidth = strideWidth == 1 ? windowSpanWidth : (windowSpanWidth + 1)/strideWidth;
		int windowHeight = strideHeight == 1 ? windowSpanHeight : (windowSpanHeight + 1)/strideHeight;
		for (Image subImage : getChannelConcatImages()) {
			subImage.populateIm2col2(data, startIndex, filterHeight, filterWidth, strideHeight, strideWidth, channels);
			startIndex = startIndex + examples * subImage.getChannels() * windowWidth * windowHeight;
		}
	}


	@Override
	public Image dup() {
		float[] dataDup = new float[data.length];
		System.arraycopy(data, 0, dataDup, 0, dataDup.length);
		return new MultiChannelImage(dataDup, channels, height, width, paddingHeight, paddingWidth, examples);
	}
	
	@Override
	public Image softDup() {
		return new MultiChannelImage(data, channels, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		for (int i = 0; i < getDataLength(); i++) {
			if (data == null) {
				System.out.println(closeStackTrace);
			}
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
	public Image getChannels(int channelRangeStart, int channelRangeEnd) {
		List<Image> channelConcatImages = getChannelConcatImages();
		for (Image image : channelConcatImages) {
			if (image.getChannels() != 1) {
				throw new IllegalStateException();
			}
		}
		List<Image> subImages = channelConcatImages.subList(channelRangeStart, channelRangeEnd);
		return new ChannelConcatImage(subImages, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public void close() {
		this.data = null;
		 ByteArrayOutputStream os = new ByteArrayOutputStream(); PrintWriter s = new
					PrintWriter(os); new RuntimeException().printStackTrace(s); s.flush();
					 s.close(); 
					 closeStackTrace = os.toString();
	}
}
