package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class TrailingActivationFunctionDirectedComponentChainImpl
 implements TrailingActivationFunctionDirectedComponentChain<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>>{

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components;
	private DifferentiableActivationFunctionComponent finalDifferentiableActivationFunctionComponent;
	private DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>> precedingChain; 
	
	public DifferentiableActivationFunctionComponent getFinalComponent() {
		return finalDifferentiableActivationFunctionComponent;
	}

	public TrailingActivationFunctionDirectedComponentChainImpl(List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.components = new ArrayList<>();
		this.components.addAll(components);
		List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decomposedList = new ArrayList<>();
		for (ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> component : decompose()) {
			decomposedList.add(component);
		}
		if (components.isEmpty()) {
			throw new IllegalArgumentException("Component list must contain at least one component");
		} else {
			ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> finalComponent = decomposedList.get(decomposedList.size() - 1);
			if (finalComponent instanceof DifferentiableActivationFunctionComponent) {
				finalDifferentiableActivationFunctionComponent = (DifferentiableActivationFunctionComponent)finalComponent;
				decomposedList.remove(decomposedList.size() - 1);
				this.precedingChain = new DefaultDirectedComponentChainImpl<ChainableDirectedComponentActivation<NeuronsActivation>>(decomposedList);
				
			} else {
				throw new IllegalArgumentException("Decomposed component list must end with a differentiable activation function component");
			}
		}
		
	}

	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponents() {
		return components;
	}

	@Override
	public TrailingActivationFunctionDirectedComponentChainActivation forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext context) {
		
		DirectedComponentChainActivation<NeuronsActivation, ChainableDirectedComponentActivation<NeuronsActivation>> precedingChainActivation = precedingChain.forwardPropagate(input, context);
		DifferentiableActivationFunctionActivation activationFunctionActivation = finalDifferentiableActivationFunctionComponent.forwardPropagate(precedingChainActivation.getOutput(), new NeuronsActivationContext() {

			/**
			 * Default serialization id.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public MatrixFactory getMatrixFactory() {
				return context.getMatrixFactory();
			}});
		
		//activationFunctionActivation.getInput().close();
		
		return new TrailingActivationFunctionDirectedComponentChainActivationImpl(precedingChainActivation, activationFunctionActivation);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return directedComponentsContext;
	}

	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
		List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> allComponents = new ArrayList<>();
		allComponents.addAll(components);
		return new DefaultDirectedComponentChainImpl<>(allComponents).decompose();
	}
}
