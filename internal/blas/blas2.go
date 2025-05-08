package blas

import "github.com/tab58/go-optimize/internal/linalg"

func GEMV(alpha float64, A linalg.Matrix, x linalg.Vector, beta float64, y linalg.Vector) {
	if A.Rows() != x.Len() {
		panic(linalg.ErrDimensionMismatch)
	}
	if A.Cols() != y.Len() {
		panic(linalg.ErrDimensionMismatch)
	}

	Ai := linalg.NewVector(A.Cols())
	for i := range A.Rows() {
		A.GetRow(i, Ai) // tmpA = A[i]
		Ax := DOT(Ai, x)
		y[i] = beta*y[i] + alpha*Ax
	}
}
