package blas

import (
	"math"

	"github.com/tab58/go-optimize/internal/linalg"
)

// SWAP swaps the elements of x and y.
func SWAP(x, y linalg.Vector) {
	if len(x) != len(y) {
		panic(linalg.ErrDimensionMismatch)
	}
	for i := range x {
		x[i], y[i] = y[i], x[i]
	}
}

// SCAL scales the elements of x by alpha = x = alpha * x
func SCAL(alpha float64, x linalg.Vector) {
	for i := range x {
		x[i] *= alpha
	}
}

// COPY copies the elements of x into y.
func COPY(x linalg.Vector, y linalg.Vector) {
	if len(x) != len(y) {
		panic(linalg.ErrDimensionMismatch)
	}
	copy(y, x)
}

// AXPY adds alpha times x to y: y = y + alpha * x
func AXPY(alpha float64, x linalg.Vector, y linalg.Vector) {
	if len(x) != len(y) {
		panic(linalg.ErrDimensionMismatch)
	}
	for i := range x {
		y[i] += alpha * x[i]
	}
}

// CPSC copies the elements of x into y, scaling by alpha: y = alpha * x
func CPSC(alpha float64, x linalg.Vector, y linalg.Vector) {
	if len(x) != len(y) {
		panic(linalg.ErrDimensionMismatch)
	}
	for i := range x {
		y[i] = alpha * x[i]
	}
}

// DOT returns the dot product of x and y.
func DOT(x, y linalg.Vector) float64 {
	if len(x) != len(y) {
		panic(linalg.ErrDimensionMismatch)
	}
	// TODO: use Kahan summation algorithm?
	sum := 0.0
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

// Hypot returns the square root of the sum of the squares of a and b in a numerically stable way.
func Hypot(a, b float64) float64 {
	if a == 0 && b == 0 {
		return 0
	}
	x := math.Abs(a)
	y := math.Abs(b)
	t := math.Min(x, y)
	u := math.Max(x, y)
	t = t / u
	return u * math.Sqrt(1+t*t)
}

// NRM2 returns the Euclidean norm of x (2-norm).
func NRM2(x linalg.Vector) float64 {
	if len(x) == 0 {
		return 0
	}
	sum := 0.0
	for i := range x {
		sum = Hypot(sum, x[i])
	}
	return sum
}

// ASUM returns the sum of the absolute values of the elements of x (L1 norm).
func ASUM(x linalg.Vector) float64 {
	if len(x) == 0 {
		return 0
	}
	sum := 0.0
	c := 0.0
	for i := range x {
		v := math.Abs(x[i])
		y := v + c
		t := sum + y
		tmp := t - sum
		c = tmp - y
		sum = t
	}
	return sum
}

// IAMAX returns the index of the element of x with the maximum absolute value.
func IAMAX(x linalg.Vector) int {
	if len(x) == 0 {
		panic(linalg.ErrIndexOutOfBounds)
	}
	max := math.Inf(-1)
	index := 0
	for i := range x {
		tmp := math.Abs(x[i])
		if tmp > max {
			max = tmp
			index = i
		}
	}
	return index
}

// Sign returns the sign of x (signum function).
func Sign(x float64) float64 {
	if x == 0 || math.IsNaN(x) {
		return x
	}
	if x > 0 {
		return 1
	}
	return -1
}

func ROTG(a, b float64) (c, s, r float64) {
	// Based on Algorithm 4 from "Discontinuous Plane Rotations
	// and the Symmetric Eigenvalue Problem" by Anderson, 2000.
	c = 0.0
	s = 0.0
	r = 0.0

	t := 0.0
	u := 0.0

	if b == 0 {
		c = Sign(a)
		s = 0
		r = math.Abs(a)
	} else if a == 0 {
		c = 0
		s = Sign(b)
		r = math.Abs(b)
	} else if math.Abs(a) > math.Abs(b) {
		t = b / a
		u = Sign(a) * math.Sqrt(1+t*t)
		c = 1 / u
		s = t * c
		r = a * u
	} else {
		t = a / b
		u = Sign(b) * math.Sqrt(1+t*t)
		s = 1 / u
		c = t * s
		r = b * u
	}
	return c, s, r
}
