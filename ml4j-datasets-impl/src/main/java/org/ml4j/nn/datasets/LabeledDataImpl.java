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
package org.ml4j.nn.datasets;

public class LabeledDataImpl<E, L> implements LabeledData<E, L> {

	private E data;
	private L label;

	public LabeledDataImpl(E data, L label) {
		this.data = data;
		this.label = label;
	}

	@Override
	public E getData() {
		return data;
	}

	@Override
	public L getLabel() {
		return label;
	}

	@Override
	public String toString() {
		return "LabeledDataImpl [data=" + data + ", label=" + label + "]";
	}

}
