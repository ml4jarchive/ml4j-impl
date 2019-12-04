package org.ml4j.nn.axons;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class Timings {

	private static Map<TimingKey, AtomicLong> timingsMap = new HashMap<>();
	private static Map<TimingKey, AtomicInteger> timingsCounts = new HashMap<>();
	private static boolean first = true;

	public synchronized static void addTime(TimingKey timingKey, long timing) {
		AtomicLong timingValue =  timingsMap.get(timingKey);
		if (timingValue == null) {
			timingValue = new AtomicLong(timing);
			timingsMap.put(timingKey, timingValue);
			timingsCounts.put(timingKey, new AtomicInteger(1));
		} else {
			AtomicInteger timingCount =  timingsCounts.get(timingKey);

			timingValue.getAndAdd(timing);
			timingCount.getAndAdd(1);

		}
		
	}
	
	public static void printTimings() {
		
		for (Entry<TimingKey, AtomicLong> entry : timingsMap.entrySet()) {
			long avg = entry.getValue().longValue() / timingsCounts.get(entry.getKey()).longValue();
			System.out.println(entry.getKey() + ":" + avg);
		}
		for (Entry<TimingKey, AtomicLong> entry : timingsMap.entrySet()) {
			long total = entry.getValue().longValue();
			System.out.println(entry.getKey() + ":" + total);
		}
		if (first) {
			timingsMap = new HashMap<>();
			timingsCounts = new HashMap<>();
		}
		
		
		
		first = false;
	}
	
}
