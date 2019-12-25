package org.ml4j.nn.components.axons.base;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DirectedAxonsComponentActivation.
 * 
 * Encapsulates the activations from a forward propagation through a DirectedAxonsComponent
 * 
 * @author Michael Lavelle
 */
public abstract class DirectedAxonsComponentActivationBase<A extends Axons<?, ?, ?>> extends DefaultChainableDirectedComponentActivationBase<DirectedAxonsComponent<?, ?, A>> implements DirectedAxonsComponentActivation {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedAxonsComponentActivationBase.class);
	
	/**
	 * The DirectedAxonsComponent that generated this activation.
	 */
	protected DirectedAxonsComponent<?, ?, A> directedAxonsComponent;
	
	protected AxonsActivation leftToRightAxonsActivation;
	
	protected AxonsContext axonsContext;

	/**
	 * Constructor for DirectedAxonsComponentActivationBase
	 * 
	 * @param axonsComponent The DirectedAxonsComponent that generated this activation.
	 * @param leftToRightAxonsActivation The activation from the axons within the directed axons component.
	 * @param axonsContext The axons context used by the axons within the directed axons component.
	 */
	public DirectedAxonsComponentActivationBase(DirectedAxonsComponent<?, ?, A> axonsComponent, AxonsActivation leftToRightAxonsActivation, AxonsContext axonsContext) {
		super(axonsComponent, leftToRightAxonsActivation.getPostDropoutOutput());
		this.directedAxonsComponent = axonsComponent;
		this.leftToRightAxonsActivation = leftToRightAxonsActivation;
		this.axonsContext = axonsContext;
	}
	
	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		// By default, a DirectedAxonsComponentActivation cannot be decomposed into smaller components, so return a singleton list containing this component.
		return Arrays.asList(this);
	}

	@Override
	public double getAverageRegularisationCost() {
		return getTotalRegularisationCost() / output.getExampleCount();
	}

	@Override
	public DirectedAxonsComponent<?, ?, A> getAxonsComponent() {
		return directedAxonsComponent;
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DirectedAxonsComponentActivationBase");
		AxonsActivation rightToLeftAxonsActivation = directedAxonsComponent.getAxons().pushRightToLeft(gradient.getOutput(), leftToRightAxonsActivation, axonsContext);
		return createBackPropagatedGradient(rightToLeftAxonsActivation, gradient.getTotalTrainableAxonsGradients(), getAxonsGradientSupplier(rightToLeftAxonsActivation));
	}
	
	protected Supplier<AxonsGradient> getAxonsGradientSupplier(AxonsActivation rightToLeftAxonsActivation) {
		return () -> getCalculatedAxonsGradient(rightToLeftAxonsActivation).orElse(null);
	}

	protected abstract Optional<AxonsGradient> getCalculatedAxonsGradient(AxonsActivation inboundGradientActivation);
	
	protected abstract DirectedComponentGradientImpl<NeuronsActivation> createBackPropagatedGradient(AxonsActivation rightToLeftAxonsActivation, List<Supplier<AxonsGradient>> previousAxonsGradients, Supplier<AxonsGradient> axonsGradientSupplier);

}
