package org.ml4j.nn.components.builders;

import org.junit.Before;
import org.junit.Test;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.DefaultComponents3DGraphBuilderFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.sessions.ComponentGraphBuilderSession;
import org.ml4j.nn.sessions.ComponentGraphBuilderSessionImpl;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

public class MyTest {

	@Mock
	private YOLOv2WeightsLoader weightsLoader;

	@Mock
	private InceptionV4WeightsLoader weightsLoader2;
	
	
	@Mock
	private DirectedComponentsContext mockDirectedComponentsContext;

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
	}

	@Test
	public void testA() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		Components3DGraphBuilderFactory<ComponentMetadata> graphBuilderFactory = new DefaultComponents3DGraphBuilderFactory<>(
				b);

		YOLOv2Definition definition = new YOLOv2Definition(weightsLoader);

		InitialComponents3DGraphBuilder<ComponentMetadata> start = graphBuilderFactory
				.createInitialComponents3DGraphBuilder(new Neurons3D(608, 608, 3, false),
						mockDirectedComponentsContext);

		definition.createComponentGraph(start, b);
	}

	@Test
	public void testB() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		YOLOv2Definition definition = new YOLOv2Definition(weightsLoader);

		InitialComponents3DGraphBuilder<ComponentMetadata> start = session.startWith(definition);

		definition.createComponentGraph(start, b);
	}
	

	@Test
	public void testC() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		YOLOv2Definition definition = new YOLOv2Definition(weightsLoader);

		definition.createComponentGraph(session).withMaxPoolingAxons("maxPooling").withFilterSize(10, 10).withSamePadding();
	}
	
	@Test
	public void testG() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		session.startWith3DNeurons(new Neurons3D(28, 28, 3, false)).withSkipConnection()
			.withActivationFunction("relu", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), 
					new ActivationFunctionProperties())
			.endSkipConnection("skip_connection");
	}
	
	@Test
	public void testH() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		session.startWith3DNeurons(new Neurons3D(28, 28, 3, false)).withSkipConnection()
			.withFullyConnectedAxons("fullyConnected").withConnectionToNeurons(new Neurons3D(20, 20, 6, false))
			.endSkipConnection("skip_connection");
	}
	
	@Test
	public void testI() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		session.startWith3DNeurons(new Neurons3D(28, 28, 3, false)).withSkipConnection()
			.withFullyConnectedAxons("fullyConnected").withConnectionToNeurons(new Neurons(400, false))
			.endSkipConnection("skip_connection").withFullyConnectedAxons("fullyConnected2")
			.withConnectionToNeurons(new Neurons(20, false)).withActivationFunction("relu", ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), 
					new ActivationFunctionProperties());
	}
	
	@Test
	public void testD() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		InceptionV4StemDefinition definition = new InceptionV4StemDefinition(weightsLoader2);

		definition.createComponentGraph(session);
	}
	
	@Test
	public void testE() {

		ComponentMetadataFactory b = new ComponentMetadataFactory();

		ComponentGraphBuilderSession<ComponentMetadata> session = new ComponentGraphBuilderSessionImpl<>(b,
				mockDirectedComponentsContext);

		InceptionV4TailDefinition definition = new InceptionV4TailDefinition(weightsLoader2);

		definition.createComponentGraph(session);
	}
	

}
