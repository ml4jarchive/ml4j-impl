package org.ml4j.nn.activationfunctions;

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
public abstract class DifferentiableActivationFunctionActivationBase implements DifferentiableActivationFunctionActivation {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DifferentiableActivationFunctionActivationBase.class);
		
	protected NeuronsActivation input;
	protected NeuronsActivation output;
	protected DifferentiableActivationFunction activationFunction;
	
	/**
	 * @param input The input to the activation function component
	 * @param output The output from the activation function component
	 */
	public DifferentiableActivationFunctionActivationBase(DifferentiableActivationFunction activationFunction, NeuronsActivation input, NeuronsActivation output) {
		this.input = input;
		this.output = output;
		this.activationFunction = activationFunction;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}
	
	@Override
	public NeuronsActivation getOutput() {
		return output;
	}
	
	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
