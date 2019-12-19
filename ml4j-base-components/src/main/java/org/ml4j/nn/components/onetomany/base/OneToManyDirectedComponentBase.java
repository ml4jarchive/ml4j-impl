package org.ml4j.nn.components.onetomany.base;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class OneToManyDirectedComponentBase<A extends OneToManyDirectedComponentActivation> implements OneToManyDirectedComponent<A> {

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
