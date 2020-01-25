package org.ml4j.images;

import java.util.ArrayList;
import java.util.List;

public class MultiChannelImage extends MultiChannelImageContainer<Image> implements Image {

	public MultiChannelImage(float[] data, int channels, int height, int width, int paddingHeight, int paddingWidth) {
		super(data, channels, height, width, paddingHeight, paddingWidth, 1);
	}

	@Override
	public MultiChannelImage dup() {
		float[] dataDup = new float[data.length];
		System.arraycopy(data, 0, dataDup, 0, dataDup.length);
		return new MultiChannelImage(dataDup, channels, height, width, paddingHeight, paddingWidth);
	}

	@Override
	public MultiChannelImage softDup() {
		return new MultiChannelImage(data, channels, height, width, paddingHeight, paddingWidth);
	}

	@Override
	protected List<Image> getChannelConcatImages() {
		List<Image> channelConcatImage = new ArrayList<>();
		int sourceStartIndex = 0;
		for (int c = 0; c < channels; c++) {
			Image channelImage = new SingleChannelImage(data, sourceStartIndex, height, width, paddingHeight,
					paddingWidth);
			channelConcatImage.add(channelImage);
			sourceStartIndex = sourceStartIndex + channelImage.getDataLength();
		}
		return channelConcatImage;
	}

	@Override
	public Image getChannels(int channelRangeStart, int channelRangeEnd) {
		List<Image> channelConcatImage = getChannelConcatImages();
		for (ImageContainer<?> image : channelConcatImage) {
			if (image.getChannels() != 1) {
				throw new IllegalStateException();
			}
		}
		List<Image> subImage = channelConcatImage.subList(channelRangeStart, channelRangeEnd);
		return new ChannelConcatImage(subImage, height, width, paddingHeight, paddingWidth);
	}

	@Override
	public Images asImages() {
		return new MultiChannelImages(getData(), channels, height, width, paddingHeight, paddingWidth, examples);
	}
}
