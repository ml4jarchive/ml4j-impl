package org.ml4j.nn.components.manytoone.base;

import java.util.Date;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ManyToOneDirectedComponentAdapter<A extends ManyToOneDirectedComponentActivation>  
	implements ManyToOneDirectedComponent<A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private ManyToOneDirectedComponent<A> delegated;
	
	public ManyToOneDirectedComponentAdapter(ManyToOneDirectedComponent<A> delegated) {
		this.delegated = delegated;
	}

	@Override
	public A forwardPropagate(List<NeuronsActivation> input, DirectedComponentsContext context) {
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
	public ManyToOneDirectedComponent<A> dup() {
		return new ManyToOneDirectedComponentAdapter<A>(delegated.dup());
	}

}
