package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
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
	private DirectedComponentFactory directedComponentFactory;
	
	public DifferentiableActivationFunctionComponent getFinalComponent() {
		return finalDifferentiableActivationFunctionComponent;
	}

	public TrailingActivationFunctionDirectedComponentChainImpl(DirectedComponentFactory directedComponentFactory, List<? extends DefaultChainableDirectedComponent<?,?>> components) {
		this.components = new ArrayList<>();
		this.components.addAll(components);
		this.directedComponentFactory = directedComponentFactory;
		List<DefaultChainableDirectedComponent<?, ?>> decomposedList = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?>  component : decompose()) {
			decomposedList.add(component);
		}
		if (components.isEmpty()) {
			throw new IllegalArgumentException("Component list must contain at least one component");
		} else {
			ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> finalComponent = decomposedList.get(decomposedList.size() - 1);
			if (finalComponent.getComponentType() == DirectedComponentType.ACTIVATION_FUNCTION) {
				finalDifferentiableActivationFunctionComponent = (DifferentiableActivationFunctionComponent)finalComponent;
				decomposedList.remove(decomposedList.size() - 1);
				this.precedingChain = directedComponentFactory.createDirectedComponentChain(decomposedList);
				
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
		return directedComponentFactory.createDirectedComponentChain(allComponents).decompose();
	}
	
	@Override
	public TrailingActivationFunctionDirectedComponentChain<DefaultChainableDirectedComponent<?, ?>> dup() {
		
		List<DefaultChainableDirectedComponent<?, ?>> dupComponents
			= components.stream().map(c -> c.dup()).collect(Collectors.toList());
		
		return new TrailingActivationFunctionDirectedComponentChainImpl(dupComponents, finalDifferentiableActivationFunctionComponent.dup(), 
				precedingChain.dup());
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN;
	}
	
}
