package org.ml4j.images;

import java.util.ArrayList;
import java.util.List;

public class ChannelConcatImages extends ChannelConcatImageContainer<Images> implements Images {

	public ChannelConcatImages(List<Images> channelConcatImages, int height, int width, int paddingHeight,
			int paddingWidth, int examples) {
		super(channelConcatImages, height, width, paddingHeight, paddingWidth, examples);
		for (Images im : channelConcatImages) {
			if (im.getExamples() != examples) {
				throw new IllegalArgumentException();
			}
		}
	}

	@Override
	public ChannelConcatImages softDup() {
		List<Images> dups = new ArrayList<>();
		for (Images image : channelConcatImages) {
			dups.add(image.softDup());
		}
		return new ChannelConcatImages(dups, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public ChannelConcatImages dup() {
		List<Images> dups = new ArrayList<>();
		for (Images image : channelConcatImages) {
			dups.add(image.dup());
		}
		return new ChannelConcatImages(dups, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public ChannelConcatImages getChannels(int channelRangeStart, int channelRangeEnd) {
		for (Images image : channelConcatImages) {
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
