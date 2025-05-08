package linalg

type Vector []float64

func NewVector(n int) Vector {
	return make([]float64, n)
}

func (v Vector) Zero() {
	for i := range v {
		v[i] = 0
	}
}

func (v Vector) Len() int {
	return len(v)
}

// Set sets all elements of the vector to the given value.
func (v Vector) Set(value float64) Vector {
	for i := range v {
		v[i] = value
	}
	return v
}
