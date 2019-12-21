package org.ml4j.nn.components.manytoone.base;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ManyToOneDirectedComponentBase<A extends ManyToOneDirectedComponentActivation> implements ManyToOneDirectedComponent<A> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(ManyToOneDirectedComponentBase.class);
	
	/**
	 * Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;
	
	protected PathCombinationStrategy pathCombinationStrategy;
	
	public ManyToOneDirectedComponentBase(PathCombinationStrategy pathCombinationStrategy) {
		this.pathCombinationStrategy = pathCombinationStrategy;
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.MANY_TO_ONE;
	}

}
