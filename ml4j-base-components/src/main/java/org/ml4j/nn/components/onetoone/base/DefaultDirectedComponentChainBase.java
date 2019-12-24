package org.ml4j.nn.components.onetoone.base;



import java.util.List;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;

/**
 * Default base class for implementations of DefaultDirectedComponentChain.
 * 
 * Encapsulates a sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultDirectedComponentChainBase 
		extends DefaultDirectedComponentChainBaseParent<DefaultChainableDirectedComponent<?, ?>, DefaultChainableDirectedComponentActivation, DefaultDirectedComponentChainActivation> 
		implements DefaultDirectedComponentChain {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainBase(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		super(sequentialComponents);
	}

}
