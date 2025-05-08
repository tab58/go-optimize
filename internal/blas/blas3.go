package blas

import "github.com/tab58/go-optimize/internal/linalg"

// GEMM performs the matrix-matrix multiplication C = alpha * A * B + beta * C
// TODO: this is a naive implementation and can be improved
func GEMM(alpha float64, A linalg.Matrix, B linalg.Matrix, beta float64, C linalg.Matrix) {
	if A.Cols() != B.Rows() {
		panic(linalg.ErrDimensionMismatch)
	}
	if A.Rows() != C.Rows() {
		panic(linalg.ErrDimensionMismatch)
	}
	if B.Cols() != C.Cols() {
		panic(linalg.ErrDimensionMismatch)
	}

	for i := range A.Rows() {
		for j := range B.Cols() {
			sum := 0.0
			for k := range A.Cols() {
				ai := A.Get(i, k)
				bj := B.Get(k, j)
				sum += ai * bj
			}
			sum *= alpha
			cij := C.Get(i, j)
			C.Set(i, j, beta*cij+sum)
		}
	}
}
