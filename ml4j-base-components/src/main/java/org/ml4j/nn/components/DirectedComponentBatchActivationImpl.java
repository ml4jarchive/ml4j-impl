package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;

public class DirectedComponentBatchActivationImpl<I, A extends DirectedComponentActivation<I, I>> implements DirectedComponentBatchActivation<I, A> {

	private List<A> activations;
	private List<I> output;
	
	public DirectedComponentBatchActivationImpl(List<A> activations) {
		this.activations = activations;
		this.output = activations.stream().map(DirectedComponentActivation::getOutput).collect(Collectors.toList());
	}
	
	@Override
	public List<A> getActivations() {
		return activations;
	}


	@Override
	public DirectedComponentGradient<List<I>> backPropagate(DirectedComponentGradient<List<I>> outerGradient) {
		int index = 0;
		List<Supplier<AxonsGradient>> allAxonsGradients = new ArrayList<>();
		allAxonsGradients.addAll(outerGradient.getTotalTrainableAxonsGradients());
		List<I> combinedOutput = new ArrayList<>();
		SortedMap<Integer, I> combinedOutputMap = new TreeMap<>();
		List<ActivationGradientIndex> activationGradients = new ArrayList<>();
		for (A activation : activations) {
			DirectedComponentGradient<I> grad = new DirectedComponentGradientImpl<>(outerGradient.getOutput().get(index));
			ActivationGradientIndex activationGradient = new ActivationGradientIndex(activation, grad, index);
			activationGradients.add(activationGradient);
			index++;
		}
		
		for (DirectedComponentBatchActivationImpl<I, A>.GradientIndex backPropGrad : activationGradients.parallelStream().map(a -> new GradientIndex(a.getActivation().backPropagate(a.getGradient()), a.getIndex())).collect(Collectors.toList())) {
			combinedOutputMap.put(backPropGrad.getIndex(), backPropGrad.getGradient().getOutput());
			combinedOutput.add(backPropGrad.getGradient().getOutput());
			List<Supplier<AxonsGradient>> backPropAxonsGradients = backPropGrad.getGradient().getTotalTrainableAxonsGradients();
			allAxonsGradients.addAll(backPropAxonsGradients);
		}
		combinedOutput.addAll(combinedOutputMap.values());

		return new DirectedComponentGradientImpl<>(allAxonsGradients, combinedOutput);
	}
	
	
	private class GradientIndex {
		private DirectedComponentGradient<I>  gradient;
		private int index;
		
		public GradientIndex(DirectedComponentGradient<I>  gradient, int index) {
			this.gradient = gradient;
			this.index = index;
		}

		public DirectedComponentGradient<I>  getGradient() {
			return gradient;
		}

		public int getIndex() {
			return index;
		}
	}
	
	private class ActivationGradientIndex {
		
		private A activation;
		private DirectedComponentGradient<I> gradient;
		private int index;
		
		public ActivationGradientIndex(A activation, DirectedComponentGradient<I> gradient, int index) {
			this.activation = activation;
			this.gradient = gradient;
			this.index = index;
		}
		
		public int getIndex() {
			return index;
		}

		public A getActivation() {
			return activation;
		}

		public DirectedComponentGradient<I> getGradient() {
			return gradient;
		}
	}


	@Override
	public List<I> getOutput() {
		return output;
	}
}
