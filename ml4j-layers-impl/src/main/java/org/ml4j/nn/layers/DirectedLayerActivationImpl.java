package org.ml4j.nn.layers;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.defaults.DefaultChainableDirectedComponentActivationAdapter;
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
		return componentChainActivation.decompose().stream().map(c -> 
		new DefaultChainableDirectedComponentActivationAdapter(c)).collect(Collectors.toList());
	}
}
