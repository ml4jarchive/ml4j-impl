package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
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

public class ResidualBlockLayerImpl extends AbstractFeedForwardLayer<Axons<?, ?, ?>, ResidualBlockLayerImpl> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private FeedForwardLayer<?, ?> layer1;
	private FeedForwardLayer<?, ?> layer2;
	private DirectedComponentFactory directedComponentFactory;
	private AxonsFactory axonsFactory;

	/**
	 * @param directedComponentFactory
	 * @param axonsFactory
	 * @param layer1
	 * @param layer2
	 * @param matrixFactory
	 */
	public ResidualBlockLayerImpl(String name, DirectedComponentFactory directedComponentFactory, AxonsFactory axonsFactory,
			FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2, MatrixFactory matrixFactory) {
		super(name, directedComponentFactory,
				createComponentChain(name, directedComponentFactory, axonsFactory, layer1, layer2, matrixFactory),
				matrixFactory);
		this.layer1 = layer1;
		this.layer2 = layer2;
		this.directedComponentFactory = directedComponentFactory;
		this.axonsFactory = axonsFactory;
	}

	private static DefaultDirectedComponentChain createPrecedingChain(String name, DirectedComponentFactory directedComponentFactory,
			FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2) {

		// Start with all components
		List<DefaultChainableDirectedComponent<?, ?>> allComponents = new ArrayList<>();
		allComponents.addAll(layer1.getComponents());
		allComponents.addAll(layer2.getComponents());

		// Set preceedingComponents list to have all but the last synapses
		List<DefaultChainableDirectedComponent<?, ?>> preceedingComponents = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> comp : allComponents.subList(0, allComponents.size() - 1)) {
			preceedingComponents.add(comp);
		}

		// Create an axons only component from the last synapses
		DirectedAxonsComponent<?, ?, ?> axonsComponent = directedComponentFactory.createDirectedAxonsComponent(name + ":PrimaryAxons",
				(Axons<? extends Neurons, ? extends Neurons, ?>) layer2.getPrimaryAxons());
		preceedingComponents.add(axonsComponent);

		// Return the component chain consisting of all components except the last
		// activation function
		return directedComponentFactory.createDirectedComponentChain(preceedingComponents);

	}

	private static DefaultDirectedComponentChain createComponentChain(String name, DirectedComponentFactory directedComponentFactory,
			AxonsFactory axonsFactory, FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2,
			MatrixFactory matrixFactory) {

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

			FullyConnectedAxons matchingAxons = axonsFactory.createFullyConnectedAxons(
					layer1.getPrimaryAxons().getLeftNeurons(), layer2.getPrimaryAxons().getRightNeurons(), 
					new WeightsMatrixImpl(null, new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
							Arrays.asList(Dimension.OUTPUT_FEATURE),WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS)),  null);
			DirectedAxonsComponent<Neurons, Neurons, ?> matchingComponent = directedComponentFactory
					.createDirectedAxonsComponent(name + ":MatchingAxons", (matchingAxons));
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
				.createDirectedComponentBipoleGraph(null, null, parallelChains, PathCombinationStrategy.ADDITION);

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
	public ResidualBlockLayerImpl dup() {
		return new ResidualBlockLayerImpl(name, directedComponentFactory, axonsFactory, layer1.dup(), layer2.dup(),
				matrixFactory);
	}

	@Override
	public Axons<?, ?, ?> getPrimaryAxons() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		List<DefaultChainableDirectedComponent<?, ?>> components = new ArrayList<>();
		components.addAll(componentChain.getComponents());
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
}
