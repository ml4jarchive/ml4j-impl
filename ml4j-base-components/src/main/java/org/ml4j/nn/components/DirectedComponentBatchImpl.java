package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;

public abstract class DirectedComponentBatchImpl<I, L extends DirectedComponent<I, A, C>, A extends DirectedComponentActivation<I, I>, C, C2> implements DirectedComponentBatch<I, L, DirectedComponentBatchActivation<I, A>, A,  C, C2> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private List<L> components;
	
	public DirectedComponentBatchImpl(List<L> components) {
		this.components = components;
	}
	
	@Override
	public List<L> getComponents() {
		return components;
	}

	@Override
	public DirectedComponentBatchActivation<I, A> forwardPropagate(List<I> input, C2 context) {

		
		
		int index = 0;
		List<ComponentInputContextIndex> componentInputContexts = new ArrayList<>();
		//List<A> activations = new ArrayList<>();
		for (L component : components) {
			ComponentInputContextIndex componentInputContext = new ComponentInputContextIndex(component, input.get(index), getContext(context, component, index), index);
			componentInputContexts.add(componentInputContext);
			//A activation = component.forwardPropagate(input.get(index), getContext(context, component, index));
			//activations.add(activation);
			index++;
		}
		
		List<ActivationIndex> activations = componentInputContexts.parallelStream().map(c -> new ActivationIndex(c.getComponent().forwardPropagate(c.getInput(), c.getContext()), c.getIndex())).collect(Collectors.toList());
		SortedMap<Integer, A> activationMap = new TreeMap<>();
		activations.forEach(a -> activationMap.put(a.getIndex(), a.getActivation()));
		List<A> sortedActivations = new ArrayList<>();
		sortedActivations.addAll(activationMap.values());
		return new DirectedComponentBatchActivationImpl<>(sortedActivations);
	}
	
	protected abstract C getContext(C2 context, L component, int index);
	
	private class ActivationIndex {
		private A activation;
		private int index;
		
		public ActivationIndex(A activation, int index) {
			this.activation = activation;
			this.index = index;
		}

		public A getActivation() {
			return activation;
		}

		public int getIndex() {
			return index;
		}
		
		
	}
	
	private class ComponentInputContextIndex {
		
		private L component;
		private I input;
		private C context;
		private int index;
		
		public ComponentInputContextIndex(L component, I input, C context, int index) {
			this.component = component;
			this.input = input;
			this.context = context;
			this.index = index;
		}
	
		public int getIndex() {
			return index;
		}



		public L getComponent() {
			return component;
		}

		public I getInput() {
			return input;
		}

		public C getContext() {
			return context;
		}
		
		
		
	}
}
