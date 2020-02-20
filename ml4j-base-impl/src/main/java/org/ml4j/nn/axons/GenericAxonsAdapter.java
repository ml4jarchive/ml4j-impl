package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

/**
 * Adapts an Axons instance whoses the concrete type of Axons is unknown, or is not important to a GenericAxons interface.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of neurons on the LHS of the axons.
 * @param <R> The type of neuron on the RHS of the axons.
 */
public class GenericAxonsAdapter<L extends Neurons, R extends Neurons> implements GenericAxons<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private Axons<L, R, ?> delegated;
	
	public GenericAxonsAdapter(Axons<L, R, ?> delegated) {
		this.delegated = delegated;
	}
	
	@Override
	public GenericAxons<L, R> dup() {
		return new GenericAxonsAdapter<>(delegated.dup());
	}

	@Override
	public AxonsType getAxonsType() {
		return delegated.getAxonsType();
	}

	@Override
	public L getLeftNeurons() {
		return delegated.getLeftNeurons();
	}

	@Override
	public R getRightNeurons() {
		return delegated.getRightNeurons();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return delegated.isTrainable(axonsContext);
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation input, AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		return delegated.pushLeftToRight(input, previousRightToLeftActivation, axonsContext);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation input, AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		return delegated.pushRightToLeft(input, previousLeftToRightActivation, axonsContext);
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return delegated.isSupported(format);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return delegated.optimisedFor();
	}

}
