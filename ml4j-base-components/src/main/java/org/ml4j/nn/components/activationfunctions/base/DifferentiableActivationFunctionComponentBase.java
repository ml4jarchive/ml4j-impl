package org.ml4j.nn.components.activationfunctions.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivationContext;

/**
 * 
 * 
 * @author Michael Lavelle
 */
public abstract class DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {

	/**
	 * Generated serialization id.
	 */
	private static final long serialVersionUID = -6033017517698579773L;
	
	protected DifferentiableActivationFunction activationFunction;
	protected Neurons neurons;
	
	public DifferentiableActivationFunctionComponentBase(Neurons neurons, DifferentiableActivationFunction activationFunction){
		this.activationFunction = activationFunction;
		this.neurons = neurons;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public NeuronsActivationContext getContext(DirectedComponentsContext context, int arg1) {
		return new NeuronsActivationContext() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public MatrixFactory getMatrixFactory() {
				return context.getMatrixFactory();
			}};
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.ACTIVATION_FUNCTION;
	}

	@Override
	public Neurons getInputNeurons() {
		return neurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return neurons;
	}
	
}
