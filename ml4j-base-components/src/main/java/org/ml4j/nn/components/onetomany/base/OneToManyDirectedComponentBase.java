package org.ml4j.nn.components.onetomany.base;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for OneToManyDirectedComponents, directed components which takes a single NeuronsActivation instance as input 
 * and map to many NeuronsActivation instances as output.
 * 
 * Used within component graphs where the flow through the NeuralNetwork is split into paths, eg. for skip-connections in ResNets or inception modules.
 * 
 * @author Michael Lavelle
 *
 * @param <A> The type of activation produced by this component on forward-propagation.
 */
public abstract class OneToManyDirectedComponentBase<A extends OneToManyDirectedComponentActivation> implements OneToManyDirectedComponent<A> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(OneToManyDirectedComponentBase.class);
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;


	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.ONE_TO_MANY;
	}

}
