package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContextConfigurer;
import org.ml4j.nn.axons.FullyConnectedAxonsConfig;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetoone.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.features.Dimension;

public class ResidualBlockLayerImpl<L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> extends AbstractFeedForwardLayer<A, ResidualBlockLayerImpl<L, R, A>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private FeedForwardLayer<?, ?> layer1;
	private FeedForwardLayer<A, ?> layer2;
	
	public static final String PRIMARY_AXONS_COMPONENT_NAME_SUFFIX = ":PrimaryAxons";

	/**
	 * @param directedComponentFactory
	 * @param axonsFactory
	 * @param layer1
	 * @param layer2
	 * @param matrixFactory
	 */
	public  ResidualBlockLayerImpl(String name, DirectedComponentFactory directedComponentFactory,
			FeedForwardLayer<?, ?> layer1, FeedForwardLayer<A, ?> layer2) {
		super(name, directedComponentFactory,
				createComponentChain(name, directedComponentFactory, layer1, layer2));
		this.layer1 = layer1;
		this.layer2 = layer2;
	}

	private static <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DefaultDirectedComponentChain createPrecedingChain(String name, DirectedComponentFactory directedComponentFactory,
			FeedForwardLayer<?, ?> layer1, FeedForwardLayer<A, ?> layer2) {

		// Start with all components
		List<DefaultChainableDirectedComponent<?, ?>> allComponents = new ArrayList<>();
		allComponents.addAll(layer1.getComponents());
		allComponents.addAll(layer2.getComponents());

		// Set precedingComponents list to have all but the last synapses
		List<DefaultChainableDirectedComponent<?, ?>> preceedingComponents = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> comp : allComponents.subList(0, allComponents.size() - 1)) {
			preceedingComponents.add(comp);
		}

		// Create an axons only component from the last synapses
		DirectedAxonsComponent<?, ?, ?> axonsComponent = directedComponentFactory.createDirectedAxonsComponent(name + PRIMARY_AXONS_COMPONENT_NAME_SUFFIX,
				layer2.getPrimaryAxons(), AxonsContextConfigurer.defaultConfigurer());
		preceedingComponents.add(axonsComponent);

		// Return the component chain consisting of all components except the last
		// activation function
		return directedComponentFactory.createDirectedComponentChain(preceedingComponents);

	}

	private static <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DefaultDirectedComponentChain createComponentChain(String name, DirectedComponentFactory directedComponentFactory, FeedForwardLayer<?, ?> layer1, FeedForwardLayer<A, ?> layer2) {

		// Final activation function component
		DifferentiableActivationFunctionComponent finalActivationFunctionComponent = directedComponentFactory
				.createDifferentiableActivationFunctionComponent(name + ":DifferentiableActivationFunction", layer2.getOutputNeurons(),
						layer2.getPrimaryActivationFunction());

		// Chain of components before the final activation function component
		DefaultDirectedComponentChain precedingChain = createPrecedingChain(name, directedComponentFactory, layer1, layer2);

		List<DefaultChainableDirectedComponent<?, ?>> matchingAxonsList = new ArrayList<>();

		// If the layer sizes don't match up, create axons to match the two sizes and
		// add to matchingAxonsList
		if (layer1.getInputNeuronCount() != (layer2.getOutputNeuronCount() + 1)) {

			DirectedAxonsComponent<Neurons, Neurons, ?> matchingComponent =  directedComponentFactory.createFullyConnectedAxonsComponent(name + ":MatchingAxons",
					FullyConnectedAxonsConfig.create(layer1.getPrimaryAxons().getLeftNeurons(), layer2.getPrimaryAxons().getRightNeurons()),
					new WeightsMatrixImpl(null, new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
					Arrays.asList(Dimension.OUTPUT_FEATURE),WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)), null);
			matchingAxonsList.add(matchingComponent);
		}

		// Skip connection chain, either empty or containing the matching axons
		DefaultDirectedComponentChain skipConnectionChain = directedComponentFactory
				.createDirectedComponentChain(matchingAxonsList);

		// Parallel Chains of preceding chain and skip connection
		List<DefaultChainableDirectedComponent<?, ?>> parallelChains = new ArrayList<>();
		parallelChains.add(precedingChain);
		parallelChains.add(skipConnectionChain);

		// Parallel Chain Batch of preceding chain and skip connection
		// DefaultDirectedComponentChainBatch parallelBatch =
		// directedComponentFactory.createDirectedComponentChainBatch(
		// parallelChains);

		// Parallel Chain Graph of preceding chain and skip connection
		// TODO - remove nulls
		DefaultDirectedComponentBipoleGraph parallelGraph = directedComponentFactory
				.createDirectedComponentBipoleGraph(null, null, null, parallelChains, PathCombinationStrategy.ADDITION);

		// Residual block component list is composed of the parallel chain graph
		// followed by the final activation function
		List<DefaultChainableDirectedComponent<?, ?>> residualBlockListOfComponents = Arrays.asList(parallelGraph,
				finalActivationFunctionComponent);

		// Create a DirectedComponentChain from the list of components, that has an
		// activation function as the final component
		return new TrailingActivationFunctionDirectedComponentChainImpl(directedComponentFactory,
				residualBlockListOfComponents);
	}

	@Override
	public int getInputNeuronCount() {
		return layer1.getInputNeuronCount();
	}

	@Override
	public int getOutputNeuronCount() {
		return layer2.getOutputNeuronCount();
	}

	@Override
	public DifferentiableActivationFunction getPrimaryActivationFunction() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public NeuronsActivation getOptimalInputForOutputNeuron(int outpuNeuronIndex,
			DirectedLayerContext directedLayerContext) {
		return layer2.getOptimalInputForOutputNeuron(outpuNeuronIndex, directedLayerContext);
	}

	@Override
	public ResidualBlockLayerImpl<L, R, A> dup(DirectedComponentFactory directedComponentFactory) {
		return new ResidualBlockLayerImpl<>(name, directedComponentFactory, layer1.dup(directedComponentFactory), layer2.dup(directedComponentFactory));
	}

	@Override
	public A getPrimaryAxons() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		List<DefaultChainableDirectedComponent<?, ?>> components = new ArrayList<>();
		components.addAll(this.trailingActivationFunctionDirectedComponentChain.getComponents());
		return components;
	}

	@Override
	public Neurons getInputNeurons() {
		return layer1.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return layer1.getOutputNeurons();
	}

	@Override
	protected String getPrimaryAxonsComponentName() {
		return name + PRIMARY_AXONS_COMPONENT_NAME_SUFFIX;
	}
}
