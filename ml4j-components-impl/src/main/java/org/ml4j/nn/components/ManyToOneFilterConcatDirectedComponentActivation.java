package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.images.Image;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ManyToOneFilterConcatDirectedComponentActivation
		extends ManyToOneDirectedComponentActivation<NeuronsActivation>
		implements DirectedComponentActivation<List<NeuronsActivation>, NeuronsActivation> {

	private static final Logger LOGGER = LoggerFactory
			.getLogger(ManyToOneFilterConcatDirectedComponentActivation.class);

	private int[] boundaries;

	public ManyToOneFilterConcatDirectedComponentActivation(NeuronsActivation output, int inputCount,
			int[] boundaries) {
		super(output, inputCount);
		this.boundaries = boundaries;

	}

	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {

		LOGGER.debug("Splitting gradient for many to one filter");

		List<NeuronsActivation> outputs = new ArrayList<>();

		NeuronsActivation outerGradientOutputActivations = outerGradient.getOutput();
		if (!(outerGradientOutputActivations instanceof ImageNeuronsActivation)) {
			throw new IllegalStateException();
		}
		
		ImageNeuronsActivation act = (ImageNeuronsActivation)outerGradientOutputActivations;
		Image img = act.getImage();
		int height = img.getHeight();
		int width = img.getWidth();
		
		for (int i = 0; i < boundaries.length; i++) {
			boundaries[i] = boundaries[i] / (height * width);
		}
		
		/*
		for (int i = 0; i < boundaries.length; i++) {
			int boundary = boundaries[i];
			NeuronsActivation subActivation = null;
			if (i == 0) {
				subActivation = outerGradientOutputActivations.filterActivationsByFeatureIndexRange(0, boundary);
			} else {
				subActivation = outerGradientOutputActivations.filterActivationsByFeatureIndexRange(boundaries[i - 1],
						boundary);
			}
			outputs.add(subActivation);
		}
		*/
		
		for (int i = 0; i < boundaries.length; i++) {
			int boundary = boundaries[i];
			NeuronsActivation subActivation = null;
			if (i == 0) {
				Image channelImage = img.getChannels(0, boundary);
				//subActivation = outerGradientOutputActivations.filterActivationsByFeatureIndexRange(0, boundary);
				subActivation = new ImageNeuronsActivationImpl(new Neurons3D(width, height, channelImage.getChannels(), false), channelImage, act.getFeatureOrientation(), act.isImmutable());
			} else {
				Image channelImage = img.getChannels(boundaries[i - 1], boundary);
				subActivation = new ImageNeuronsActivationImpl(new Neurons3D(width, height, channelImage.getChannels(), false), channelImage, act.getFeatureOrientation(), act.isImmutable());
				//subActivation = outerGradientOutputActivations.filterActivationsByFeatureIndexRange(boundaries[i - 1],
				//		boundary);
			}
			outputs.add(subActivation);
		}

		LOGGER.debug("End splitting gradient for many to one filter");

		return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(), outputs);
	}

}
