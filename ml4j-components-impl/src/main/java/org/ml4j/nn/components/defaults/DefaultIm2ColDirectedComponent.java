package org.ml4j.nn.components.defaults;

import java.util.Arrays;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultIm2ColDirectedComponent implements ChainableDirectedComponent<NeuronsActivation, DefaultIm2ColDirectedComponentActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	private Axons3DConfig config;
	
	public DefaultIm2ColDirectedComponent(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
	}
	
	
	@Override
	public DefaultIm2ColDirectedComponentActivation forwardPropagate(NeuronsActivation input, DirectedComponentsContext context) {
		
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();
		
		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;
		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());
		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());
		
		System.out.println(inputWidthWithPadding);
		System.out.println(outputWidth);
		System.out.println(filterWidth);
		
		Neurons3D neurons = new Neurons3D(299, 299, 3, false);
	
		ImageNeuronsActivation imageNeuronsActivation = input.asImageNeuronsActivation(leftNeurons);
		Matrix im2Col = imageNeuronsActivation.im2Col(context.getMatrixFactory(), filterHeight, filterWidth, config.getStrideHeight(), config.getStrideWidth(), 
				config.getPaddingHeight(), config.getPaddingWidth());
		ImageNeuronsActivation output = new ImageNeuronsActivationImpl(im2Col, neurons, input.getFeatureOrientation(), input.isImmutable());
		return new DefaultIm2ColDirectedComponentActivation(output);
	}


	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
		return Arrays.asList(this);
	}


	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int index) {
		return context;
	}

	
}
