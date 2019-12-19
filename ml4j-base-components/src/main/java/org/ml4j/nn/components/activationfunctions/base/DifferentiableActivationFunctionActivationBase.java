package org.ml4j.nn.components.activationfunctions.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DifferentiableActivationFunctionActivation.
 * 
 * Encapsulates the activations from an activation through a DifferentiableActivationFunctionComponent
 * 
 * @author Michael Lavelle
 */
public abstract class DifferentiableActivationFunctionActivationBase extends DefaultChainableDirectedComponentActivationBase implements DifferentiableActivationFunctionActivation {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DifferentiableActivationFunctionActivationBase.class);
		
	protected NeuronsActivation input;
	
	/**
	 * @param input The input to the activation function component
	 * @param output The output from the activation function component
	 */
	public DifferentiableActivationFunctionActivationBase(NeuronsActivation input, NeuronsActivation output) {
		super(output);
		this.input = input;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}
}
