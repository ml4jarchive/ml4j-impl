/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.axons;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Simple implementation of AxonsContext.
 * 
 * @author Michael Lavelle
 */
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.FreezeableNeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationContextImpl;

public class AxonsContextImpl extends NeuronsActivationContextImpl implements AxonsContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private boolean localFreezeOut;

	private float regularisationLambda;

	private float leftHandInputDropoutKeepProbability;

	private List<FreezeableNeuronsActivationContext<?>> freezeOutOverrideContexts;

	private String axonsName;

	/**
	 * Construct a new AxonsContext.
	 * 
	 * @param matrixFactory The MatrixFactory we configure for this context
	 * @param withFreezeOut Whether to freeze out these Axons.
	 */
	public AxonsContextImpl(String axonsName, MatrixFactory matrixFactory, boolean isTrainingContext,
			boolean withFreezeOut) {
		super(matrixFactory, isTrainingContext);
		this.leftHandInputDropoutKeepProbability = 1f;
		this.localFreezeOut = withFreezeOut;
		this.freezeOutOverrideContexts = new ArrayList<>();
		this.axonsName = axonsName;
	}

	@Override
	public float getLeftHandInputDropoutKeepProbability() {
		return leftHandInputDropoutKeepProbability;
	}

	@Override
	public boolean isWithFreezeOut() {
		synchronized (freezeOutOverrideContexts) {
			if (freezeOutOverrideContexts.isEmpty()) {
				return localFreezeOut;
			} else {
				FreezeableNeuronsActivationContext<?> finalOverride = 
						freezeOutOverrideContexts.get(freezeOutOverrideContexts.size() - 1);
				if (finalOverride == this) {
					return localFreezeOut;
				} else {
					return finalOverride.isWithFreezeOut();
				}
			}
		}
	}
	
	private void tidySelf() {
		synchronized (freezeOutOverrideContexts) {
			if (freezeOutOverrideContexts.size() == 1 && freezeOutOverrideContexts.get(0) == this) {
				removeFreezeoutOverrideContext(this);
			}
		}
	}

	@Override
	public AxonsContext withFreezeOut(boolean withFreezeOut) {
		synchronized (freezeOutOverrideContexts) {
			this.localFreezeOut = withFreezeOut;
			if (!freezeOutOverrideContexts.isEmpty()) {
				if (withFreezeOut) {
					addFreezeoutOverrideContext(this);
				} else {
					removeFreezeoutOverrideContext(this);
				}
			}
			tidySelf();
			return this;
		}
	}

	@Override
	public float getRegularisationLambda() {
		return regularisationLambda;
	}

	@Override
	public AxonsContext withLeftHandInputDropoutKeepProbability(float leftHandInputDropoutKeepProbability) {
		this.leftHandInputDropoutKeepProbability = leftHandInputDropoutKeepProbability;
		return this;
	}

	@Override
	public AxonsContext withRegularisationLambda(float regularisationLambda) {
		this.regularisationLambda = regularisationLambda;
		return this;
	}

	private List<String> getFreezeOutOverrideComponentNames() {
		return freezeOutOverrideContexts.stream().map(c -> c.getOwningComponentName()).collect(Collectors.toList());
	}

	@Override
	public String toString() {
		return "AxonsContextImpl [ name='" + axonsName + "' isWithFreezeOut()=" + isWithFreezeOut()
				+ ", localFreezeOut=" + localFreezeOut + ", freezeOutOverrideComponentNames="
				+ getFreezeOutOverrideComponentNames() + ", isTrainingContext=" + isTrainingContext()
				+ ", leftHandInputDropoutKeepProbability=" + leftHandInputDropoutKeepProbability
				+ ", regularisationLambda=" + regularisationLambda + "]";
	}

	@Override
	public AxonsContext dup() {
		return new AxonsContextImpl(axonsName, getMatrixFactory(), isTrainingContext(), isWithFreezeOut())
				.withRegularisationLambda(regularisationLambda)
				.withLeftHandInputDropoutKeepProbability(leftHandInputDropoutKeepProbability);
	}

	@Override
	public String getOwningComponentName() {
		return axonsName;
	}

	@Override
	public void addFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> context) {
		synchronized (freezeOutOverrideContexts) {
			if (context.isWithFreezeOut() != this.isWithFreezeOut()) {
				removeFreezeoutOverrideContext(context);
				freezeOutOverrideContexts.add(context);
			} else {
				removeFreezeoutOverrideContext(context);
			}
			tidySelf();
		}
	}

	@Override
	public void removeFreezeoutOverrideContext(FreezeableNeuronsActivationContext<?> context) {
		synchronized (freezeOutOverrideContexts) {
			freezeOutOverrideContexts.removeIf(c -> c == context);
			tidySelf();
		}
	}
	
	@Override
	public AxonsContext asNonTrainingContext() {
		AxonsContextImpl axonsContext =  new AxonsContextImpl(axonsName, getMatrixFactory(), false, localFreezeOut);
		axonsContext.withLeftHandInputDropoutKeepProbability(leftHandInputDropoutKeepProbability);
		axonsContext.withRegularisationLambda(regularisationLambda);
		axonsContext.freezeOutOverrideContexts = this.freezeOutOverrideContexts;
		return axonsContext;
	}

	@Override
	public AxonsContext asTrainingContext() {
		AxonsContextImpl axonsContext =  new AxonsContextImpl(axonsName, getMatrixFactory(), true, localFreezeOut);
		axonsContext.withLeftHandInputDropoutKeepProbability(leftHandInputDropoutKeepProbability);
		axonsContext.withRegularisationLambda(regularisationLambda);
		axonsContext.freezeOutOverrideContexts = this.freezeOutOverrideContexts;
		return axonsContext;
	}

}
