package org.ml4j.nn.components.builders;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * An example of using a custom NeuralComponentFactory to create cusom NeuralComponents into InceptionV4 graphs.
 * 
 * Such custom components can be created in InceptionV4 graphs by a InceptionV4Definition in the same way as the default components
 * 
 * @author Michael Lavelle
 */
public class ComponentMetadataFactory implements NeuralComponentFactory<ComponentMetadata>{

	@Override
	public ComponentMetadata createAveragePoolingAxonsComponent(String name, Axons3DConfig arg2) {
		return new ComponentMetadata(name, arg2.getLeftNeurons(), arg2.getRightNeurons(), "Average Pooling Axons:" + name);
	}

	@Override
	public <N extends Neurons> ComponentMetadata createBatchNormAxonsComponent(String name, N arg0, N arg1) {
		return new ComponentMetadata(name, arg0, arg1, "Batch Norm Axons:" + name);
	}

	@Override
	public <N extends Neurons> ComponentMetadata createBatchNormAxonsComponent(String name, N arg0, N arg1, WeightsMatrix arg2, BiasMatrix arg3,
			Matrix arg4, Matrix arg5) {
		return new ComponentMetadata(name, arg0, arg1, "Batch Norm Axons:" + name);
	}

	@Override
	public ComponentMetadata createConvolutionalAxonsComponent(String name, Axons3DConfig arg2,
			WeightsMatrix arg3, BiasMatrix arg4) {
		return new ComponentMetadata(name, arg2.getLeftNeurons(), arg2.getRightNeurons(), "Convolutional Axons:" + name);
	}

	@Override
	public ComponentMetadata createConvolutionalBatchNormAxonsComponent(String name, Neurons3D arg0, Neurons3D arg1) {
		return new ComponentMetadata(name, arg0, arg1, "Convolutional Batch Norm Axons:" + name);
	}

	@Override
	public ComponentMetadata createConvolutionalBatchNormAxonsComponent(String name, Neurons3D arg0, Neurons3D arg1, WeightsMatrix arg2,
			BiasMatrix arg3, Matrix arg4, Matrix arg5) {
		return new ComponentMetadata(name, arg0, arg1 ,"Convolutional Batch Norm Axons:" + name);
	}

	@Override
	public ComponentMetadata createDifferentiableActivationFunctionComponent(String name, Neurons arg0,
			DifferentiableActivationFunction arg1) {
		return new ComponentMetadata(name, arg0, arg0, "Activation Function:" + arg1.getClass() + ":" + name);
	}
	
	@Override
	public ComponentMetadata createDifferentiableActivationFunctionComponent(String name, Neurons arg0,
			ActivationFunctionType arg1, ActivationFunctionProperties activationFunctionProperties) {
		return new ComponentMetadata(name, arg0, arg0, "Activation Function:" + arg1.getQualifiedId() + ":" + name);
	}

	@Override
	public ComponentMetadata createDirectedComponentBipoleGraph(String name, Neurons arg0, Neurons arg1,
			List<ComponentMetadata> arg2, PathCombinationStrategy arg3) {
		return new ComponentMetadata(name, arg0, arg1, "Bipole Graph with strategy:" + arg3);
	}

	@Override
	public ComponentMetadata createDirectedComponentChain(List<ComponentMetadata> arg0) {
		return new ComponentMetadata("ComponentChain", arg0.get(0).getInputNeurons(), arg0.get(arg0.size() -1).getOutputNeurons(), "Component Chain with " + arg0.size() + " components");
	}

	@Override
	public ComponentMetadata createFullyConnectedAxonsComponent(String name, AxonsConfig<Neurons, Neurons> axonsConfig, WeightsMatrix arg2, BiasMatrix arg3) {
		return new ComponentMetadata(name, axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons(), "Fully Connected Axons Component:" + name);
	}

	@Override
	public ComponentMetadata createMaxPoolingAxonsComponent(String name, Axons3DConfig arg2,
			boolean arg3) {
		return new ComponentMetadata(name, arg2.getLeftNeurons(), arg2.getRightNeurons(), "Max Pooling Axons Component:" + name);
	}

	@Override
	public <N extends Neurons> ComponentMetadata createPassThroughAxonsComponent(String name,N arg0, N arg1) {
		return new ComponentMetadata(name, arg0, arg1, "Pass through Axons Component:" + name);
	}

	@Override
	public ComponentMetadata createComponent(String name, Neurons arg0, Neurons arg1,
			NeuralComponentType componentType) {
		return new ComponentMetadata(name, arg0, arg1, "Component type:" + componentType + ":" + name);
	}
	
}
