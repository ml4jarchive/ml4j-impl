package org.ml4j.images;

public class SingleChannelImages extends SingleChannelImageContainer<Images> implements Images {

	public SingleChannelImages(float[] data, int startIndex, int height, int width, int paddingHeight, int paddingWidth,
			int examples) {
		super(data, startIndex, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public SingleChannelImages dup() {
		float[] dataDup = new float[data.length];
		System.arraycopy(data, 0, dataDup, 0, dataDup.length);
		return new SingleChannelImages(dataDup, startIndex, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public SingleChannelImages softDup() {
		return new SingleChannelImages(data, startIndex, height, width, paddingHeight, paddingWidth, examples);
	}

	@Override
	public SingleChannelImages getChannels(int channelRangeStart, int channelRangeEnd) {
		if (channelRangeStart == 0 && channelRangeEnd == 0) {
			return this;
		} else {
			throw new IllegalArgumentException();
		}
	}

	@Override
	public int getExamples() {
		return examples;
	}

}
