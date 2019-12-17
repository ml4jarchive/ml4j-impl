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

}
