package optim

import (
	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
)

// UpdateHessianBFGS is a rank-2 update of the Hessian using the BFGS method.
func UpdateHessianBFGS(H linalg.Matrix, y linalg.Vector, dx linalg.Vector) {
	if y.Len() != dx.Len() {
		panic(linalg.ErrDimensionMismatch)
	}
	if y.Len() != H.Rows() {
		panic(linalg.ErrDimensionMismatch)
	}

	t1 := linalg.NewVector(y.Len())
	blas.GEMV(1.0, H, dx, 0.0, t1) // t1 = Hk * dxk

	a := 1.0 / blas.DOT(y, dx)   // a = yk^T * dxk
	b := -1.0 / blas.DOT(dx, t1) // b = dxk^T * Hk * dxk

	H.AddOuterProduct(y, y, a)
	H.AddOuterProduct(t1, t1, b)
}

// UpdateHessianInverseBFGS is a rank-2 update of the Hessian inverse using the BFGS method.
func UpdateHessianInverseBFGS(N linalg.Matrix, y linalg.Vector, dx linalg.Vector) {
	if y.Len() != dx.Len() {
		panic(linalg.ErrDimensionMismatch)
	}
	if y.Len() != N.Rows() {
		panic(linalg.ErrDimensionMismatch)
	}
	if dx.Len() != N.Cols() {
		panic(linalg.ErrDimensionMismatch)
	}

	// fmt.Printf("BFGS update N: %v\n", N)
	// fmt.Printf("BFGS update dx: %v\n", dx)
	// fmt.Printf("BFGS update y: %v\n", y)

	t1 := linalg.NewVector(y.Len())
	blas.GEMV(1.0, N, y, 0.0, t1) // t1 = Nk * yk

	a := blas.DOT(dx, y)
	b := blas.DOT(y, t1)
	c := 1.0 / a
	d := (1.0 + b/a) * c

	// fmt.Printf("BFGS update a: %v\n", a)
	// fmt.Printf("BFGS update b: %v\n", b)
	// fmt.Printf("BFGS update c: %v\n", c)
	// fmt.Printf("BFGS update d: %v\n", d)

	N.AddOuterProduct(dx, dx, d)
	N.AddOuterProduct(dx, t1, -c)
	N.AddOuterProduct(t1, dx, -c)

	// fmt.Printf("N after update: %v\n", N)
}
