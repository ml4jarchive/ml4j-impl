package org.ml4j.jblas;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public class DefaultFloatArrayFactory implements FloatArrayFactory {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	private static Map<Integer, float[]> cache = new HashMap<>();
	private static Map<Integer, Boolean> existed = new HashMap<>();
	
	private static int MAX_LENGTH = 1800000000;
	
	private static AtomicLong currentLength = new AtomicLong(0);

	@Override
	public float[] createFloatArray(int length) {
		//currentLength.getAndAdd(length);
		return new float[length]; 
	}
	
	 public static void addMatrixToCache(float[] array) {
			//currentLength.getAndAdd(-array.length);
	 }
}
