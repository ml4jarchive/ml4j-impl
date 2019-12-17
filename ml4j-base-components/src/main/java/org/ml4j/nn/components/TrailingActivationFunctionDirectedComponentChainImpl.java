package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class TrailingActivationFunctionDirectedComponentChainImpl
 implements TrailingActivationFunctionDirectedComponentChain<DefaultChainableDirectedComponent<?, ?>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private List<DefaultChainableDirectedComponent<?, ?>> components;
	private DifferentiableActivationFunctionComponent finalDifferentiableActivationFunctionComponent;
	private DefaultDirectedComponentChain precedingChain; 
	
	public DifferentiableActivationFunctionComponent getFinalComponent() {
		return finalDifferentiableActivationFunctionComponent;
	}

	public TrailingActivationFunctionDirectedComponentChainImpl(List<? extends DefaultChainableDirectedComponent<?,?>> components) {
		this.components = new ArrayList<>();
		this.components.addAll(components);
		List<DefaultChainableDirectedComponent<?, ?>> decomposedList = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?>  component : decompose()) {
			decomposedList.add(component);
		}
		if (components.isEmpty()) {
			throw new IllegalArgumentException("Component list must contain at least one component");
		} else {
			ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> finalComponent = decomposedList.get(decomposedList.size() - 1);
			if (finalComponent instanceof DifferentiableActivationFunctionComponent) {
				finalDifferentiableActivationFunctionComponent = (DifferentiableActivationFunctionComponent)finalComponent;
				decomposedList.remove(decomposedList.size() - 1);
				this.precedingChain = new DefaultDirectedComponentChainImpl(decomposedList);
				
			} else {
				throw new IllegalArgumentException("Decomposed component list must end with a differentiable activation function component");
			}
		}
		
	}
	
	

	private  TrailingActivationFunctionDirectedComponentChainImpl(
			List<DefaultChainableDirectedComponent<?, ?>> components,
			DifferentiableActivationFunctionComponent finalDifferentiableActivationFunctionComponent,
			DefaultDirectedComponentChain precedingChain) {
		this.components = components;
		this.finalDifferentiableActivationFunctionComponent = finalDifferentiableActivationFunctionComponent;
		this.precedingChain = precedingChain;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		return components;
	}

	@Override
	public TrailingActivationFunctionDirectedComponentChainActivation forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext context) {
		
		DefaultDirectedComponentChainActivation precedingChainActivation = precedingChain.forwardPropagate(input, context);
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
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		List<DefaultChainableDirectedComponent<?, ?>> allComponents = new ArrayList<>();
		allComponents.addAll(components);
		return new DefaultDirectedComponentChainImpl(allComponents).decompose();
	}
	
	@Override
	public TrailingActivationFunctionDirectedComponentChain<DefaultChainableDirectedComponent<?, ?>> dup() {
		
		List<DefaultChainableDirectedComponent<?, ?>> dupComponents
			= components.stream().map(c -> c.dup()).collect(Collectors.toList());
		
		return new TrailingActivationFunctionDirectedComponentChainImpl(dupComponents, finalDifferentiableActivationFunctionComponent.dup(), 
				precedingChain.dup());
	}
	
}
