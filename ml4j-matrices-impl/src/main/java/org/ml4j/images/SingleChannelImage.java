package org.ml4j.images;

public class SingleChannelImage extends SingleChannelImageContainer<Image> implements Image {

	public SingleChannelImage(float[] data, int startIndex, int height, int width, int paddingHeight,
			int paddingWidth) {
		super(data, startIndex, height, width, paddingHeight, paddingWidth, 1);
	}

	@Override
	public SingleChannelImage dup() {
		float[] dataDup = new float[data.length];
		System.arraycopy(data, 0, dataDup, 0, dataDup.length);
		return new SingleChannelImage(dataDup, startIndex, height, width, paddingHeight, paddingWidth);
	}

	@Override
	public SingleChannelImage softDup() {
		return new SingleChannelImage(data, startIndex, height, width, paddingHeight, paddingWidth);
	}

	@Override
	public SingleChannelImage getChannels(int channelRangeStart, int channelRangeEnd) {
		if (channelRangeStart == 0 && channelRangeEnd == 0) {
			return this;
		} else {
			throw new IllegalArgumentException();
		}
	}
}
