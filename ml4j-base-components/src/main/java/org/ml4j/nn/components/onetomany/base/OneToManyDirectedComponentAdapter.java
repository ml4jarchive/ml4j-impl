package org.ml4j.nn.components.onetomany.base;

import java.util.Date;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;

public class OneToManyDirectedComponentAdapter<A extends OneToManyDirectedComponentActivation>  
	implements OneToManyDirectedComponent<A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private OneToManyDirectedComponent<A> delegated;
	
	public OneToManyDirectedComponentAdapter(OneToManyDirectedComponent<A> delegated) {
		this.delegated = delegated;
	}

	@Override
	public A forwardPropagate(NeuronsActivation input, DirectedComponentsContext context) {
		long startTime = new Date().getTime();
		A activation =  delegated.forwardPropagate(input, context);
		long endTime = new Date().getTime();
		long timeTaken = endTime - startTime;
		DefaultChainableDirectedComponentAdapter.addTime(timeTaken, delegated.getClass().getSimpleName());
		return activation;
	}

	@Override
	public DirectedComponentType getComponentType() {
		return delegated.getComponentType();
	}

	@Override
	public OneToManyDirectedComponent<A> dup() {
		return new OneToManyDirectedComponentAdapter<A>(delegated.dup());
	}

}
