/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.axons;

import org.ml4j.Matrix;

/**
 * Default implementation of AxonsDropoutMask
 * 
 * @author Michael Lavelle
 *
 */
public class AxonsDropoutMaskImpl implements AxonsDropoutMask {

	private Matrix dropoutMask;
	private AxonsDropoutMaskType type;

	public AxonsDropoutMaskImpl(Matrix dropoutMask, AxonsDropoutMaskType type) {
		this.dropoutMask = dropoutMask;
		this.type = type;
	}

	@Override
	public Matrix getDropoutMask() {
		return dropoutMask;
	}

	@Override
	public AxonsDropoutMaskType getType() {
		return type;
	}

}
