package org.ml4j.images;

import java.util.ArrayList;
import java.util.List;

public class MultiChannelImages extends MultiChannelImageContainer<Images> implements Images {

	public MultiChannelImages(float[] data, int startIndex, int channels, int height, int width, int paddingHeight, int paddingWidth,
			int examples) {
		super(data, startIndex, channels, height, width, paddingHeight, paddingWidth, examples);
	}
	
	public MultiChannelImages(float[] data, int channels, int height, int width, int paddingHeight, int paddingWidth,
			int examples) {
		super(data, 0, channels, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public MultiChannelImages dup() {
		float[] dataDup = new float[getDataLength()];
		populateData(dataDup, 0);
		return new MultiChannelImages(dataDup, 0, channels, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public MultiChannelImages softDup() {
		return new MultiChannelImages(data, startIndex, channels, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	protected List<Images> getChannelConcatImages() {
		List<Images> channelConcatImages = new ArrayList<>();
		int sourceStartIndex = this.startIndex;
		for (int c = 0; c < channels; c++) {
			Images channelImages = new SingleChannelImages(data, sourceStartIndex, height, width, paddingHeight,
					paddingWidth, examples);
			channelConcatImages.add(channelImages);
			sourceStartIndex = sourceStartIndex + channelImages.getDataLength();
		}
		return channelConcatImages;
	}

	@Override
	public Images getChannels(int channelRangeStart, int channelRangeEnd) {
		List<Images> channelConcatImages = getChannelConcatImages();
		for (ImageContainer<?> image : channelConcatImages) {
			if (image.getChannels() != 1) {
				throw new IllegalStateException();
			}
		}
		List<Images> subImages = channelConcatImages.subList(channelRangeStart, channelRangeEnd);
		return new ChannelConcatImages(subImages, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public int getExamples() {
		return examples;
	}
}
