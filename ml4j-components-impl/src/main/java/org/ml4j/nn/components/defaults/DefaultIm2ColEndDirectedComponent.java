package org.ml4j.nn.components.defaults;

import java.util.Arrays;
import java.util.List;

import org.ml4j.EditableMatrix;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultIm2ColEndDirectedComponent implements ChainableDirectedComponent<NeuronsActivation, DefaultIm2ColEndDirectedComponentActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	private Axons3DConfig config;
	
	public DefaultIm2ColEndDirectedComponent(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig config) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
	}
	
	
	@Override
	public DefaultIm2ColEndDirectedComponentActivation forwardPropagate(NeuronsActivation input, DirectedComponentsContext context) {
		EditableMatrix m = input.getActivations(context.getMatrixFactory()).asEditableMatrix();
		int examples = input.getFeatureCount() * input.getExampleCount() / (rightNeurons.getDepth() * rightNeurons.getHeight() * rightNeurons.getWidth());
		m.reshape(rightNeurons.getDepth() * rightNeurons.getHeight() * rightNeurons.getWidth() , examples);
		ImageNeuronsActivation output = new ImageNeuronsActivationImpl(m, rightNeurons, input.getFeatureOrientation(), false);
		return new DefaultIm2ColEndDirectedComponentActivation(output);
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
