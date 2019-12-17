package org.ml4j.nn.datasets.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.UncheckedIOException;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.BatchedDataSet;
import org.ml4j.nn.datasets.BatchedDataSetImpl;
import org.ml4j.nn.datasets.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ObjectInputStreamDataSet<E> implements DataSet<E> {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(ObjectInputStreamDataSet.class);
	
	private Supplier<ObjectInputStream> objectInputStreamSupplier;
	private Class<E> cls;
	
	public ObjectInputStreamDataSet(Supplier<ObjectInputStream> objectInputStreamSupplier, Class<E> cls) {
		this.objectInputStreamSupplier = objectInputStreamSupplier; 
		this.cls = cls;
	}
	
	public ObjectInputStreamDataSet(File file, Class<E> cls) {
		this.objectInputStreamSupplier = () -> createObjectInputStream(file); 
		this.cls = cls;
	}
	
	private ObjectInputStream createObjectInputStream(File file) {
		try {
			FileInputStream fis = new FileInputStream(file);
			try {
				return new ObjectInputStream(fis);
			} catch (IOException e) {
				try {
					fis.close();
					throw new UncheckedIOException(e);
				} catch (IOException e1) {
					LOGGER.warn("Unable to close file input stream", e1);					
					throw new UncheckedIOException(e);
				}
			}

		} catch (FileNotFoundException e) {
			throw new UncheckedIOException(e);
		}
	}

	@Override
	public Stream<E> stream() {
		return StreamUtil.toStream(objectInputStreamSupplier.get(), cls);
	}

	@Override
	public BatchedDataSet<E> toBatchedDataSet(int batchSize) {
		return new BatchedDataSetImpl<E>(() -> StreamUtil.partition(stream(), batchSize));
	}

}
