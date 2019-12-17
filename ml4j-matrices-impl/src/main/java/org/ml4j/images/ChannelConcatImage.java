package org.ml4j.images;

import java.util.ArrayList;
import java.util.List;

public class ChannelConcatImage extends ChannelConcatImageContainer<Image> implements Image {

	public ChannelConcatImage(List<Image> channelConcatImages, int height, int width, int paddingHeight, int paddingWidth) {
		super(channelConcatImages, height, width, paddingHeight, paddingWidth, 1);
	}
	
	@Override
	public ChannelConcatImage softDup() {
		List<Image> dups = new ArrayList<>();
		for (Image image : channelConcatImages) {
			dups.add(image.softDup());
		}
		return new ChannelConcatImage(dups, height, width, paddingHeight, paddingWidth);
	}
	
	@Override
	public ChannelConcatImage dup() {
		List<Image> dups = new ArrayList<>();
		for (Image image : channelConcatImages) {
			dups.add(image.dup());
		}
		return new ChannelConcatImage(dups, height, width, paddingHeight, paddingWidth);
	}

	@Override
	public ChannelConcatImage getChannels(int channelRangeStart, int channelRangeEnd) {
		for (Image image : channelConcatImages) {
			if (image.getChannels() != 1) {
				throw new IllegalStateException();
			}
		}
		List<Image> subImages = channelConcatImages.subList(channelRangeStart, channelRangeEnd);
		return new ChannelConcatImage(subImages, height, width, paddingHeight, paddingWidth);
	}
}
