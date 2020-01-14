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

package org.ml4j.nn;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of BackPropagation.
 * 
 * @author Michael Lavelle
 */
public class BackPropagationImpl implements BackPropagation {

  private DirectedComponentGradient<NeuronsActivation> gradient;

  public BackPropagationImpl(DirectedComponentGradient<NeuronsActivation> gradient) {
    this.gradient = gradient;
  }

  @Override
  public DirectedComponentGradient<NeuronsActivation> getGradient() {
    return gradient;
  }
}
