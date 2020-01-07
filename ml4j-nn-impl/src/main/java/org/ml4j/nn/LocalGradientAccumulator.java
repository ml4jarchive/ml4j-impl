package org.ml4j.nn;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.Future;

import org.ml4j.nn.axons.AxonsGradient;

public class LocalGradientAccumulator implements GradientAccumulator {

	@Override
	public Optional<Future<List<AxonsGradient>>> submitCostAndGradients(CostAndGradients costAndGradients) {
		return Optional.of(new SimpleAverageAxonsGradientsFuture(costAndGradients));
	}

}
