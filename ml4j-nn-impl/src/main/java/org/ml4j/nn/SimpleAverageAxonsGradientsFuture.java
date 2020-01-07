package org.ml4j.nn;

import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.ml4j.nn.axons.AxonsGradient;

public class SimpleAverageAxonsGradientsFuture implements Future<List<AxonsGradient>>{
	
	private CostAndGradients costAndGradients;
	
	public SimpleAverageAxonsGradientsFuture(CostAndGradients costAndGradients) {
		this.costAndGradients = costAndGradients;
	}

	@Override
	public boolean cancel(boolean mayInterruptIfRunning) {
		return false;
	}

	@Override
	public boolean isCancelled() {
		return false;
	}

	@Override
	public boolean isDone() {
		return true;
	}

	@Override
	public List<AxonsGradient> get() {
		return costAndGradients.getAverageTrainableAxonsGradients();
	}

	@Override
	public List<AxonsGradient> get(long timeout, TimeUnit unit) {
		return costAndGradients.getAverageTrainableAxonsGradients();
	}

}
