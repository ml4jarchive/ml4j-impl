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

/**
 * Default config for 3D Axons.
 * 
 * @author Michael Lavelle
 *
 */
public class Axons3DConfig extends AxonsConfig {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  /**
   * The stride of these Axons.
   */
  private int stride = 1;

  public int getStride() {
    return stride;
  }

  public Axons3DConfig withStride(int stride) {
    this.stride = stride;
    return this;
  }
}
