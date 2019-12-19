package org.ml4j.nn.layers;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirectedLayerActivationImpl implements DirectedLayerActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedLayerActivationImpl.class);

	private TrailingActivationFunctionDirectedComponentChainActivation componentChainActivation;

	private DirectedLayer<?, ?> layer;
	private DirectedLayerContext layerContext;

	public DirectedLayerActivationImpl(DirectedLayer<?, ?> layer,
			TrailingActivationFunctionDirectedComponentChainActivation componentChainActivation,
			DirectedLayerContext layerContext) {
		this.componentChainActivation = componentChainActivation;
		this.layer = layer;
		this.layerContext = layerContext;
	}

	@Override
	public NeuronsActivation getOutput() {
		return componentChainActivation.getOutput();
	}

	@Override
	public DirectedLayer<?, ?> getLayer() {
		return layer;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		return componentChainActivation.backPropagate(outerGradient);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient activationGradient) {
		LOGGER.debug(
				layerContext.toString() + ":" + "Back propagating cost function gradient through layer activation....");
		return componentChainActivation.backPropagate(activationGradient);
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return componentChainActivation.decompose();
	}
}
