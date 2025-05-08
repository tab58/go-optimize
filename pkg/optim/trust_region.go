package optim

import (
	"math"

	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
)

var TOLERANCE = 1e-6
var RMAX = 1.0 // radius of the trust region
var BETA = 1.0 / 16.0
var ETA_MAX = 2.0

func ComputeTrustRegion(x0 linalg.Vector, f ObjectiveFunc, gradF gradientFunc, x1 linalg.Vector, B linalg.Matrix, Binv linalg.Matrix) {
	r0 := 0.1
	n := x0.Len()
	df := linalg.NewVector(n)
	dx := linalg.NewVector(n)
	xk := linalg.NewVector(n)
	t1 := linalg.NewVector(n)

	// setup initial values
	// gradNorm := math.Inf(1)
	dx.Zero()
	blas.COPY(x0, xk)

	rk := r0
	maxiter := 10000 // making sure this doesn't run forever
	iter := 0
	// fmt.Println("--- STARTING TRUST REGION ---")
	gradNorm := gradF(xk, f, df)
	for gradNorm > TOLERANCE && iter < maxiter {
		// fmt.Printf("gradNorm: %v\n", gradNorm)
		// fmt.Printf("iter: %v\n", iter)
		// fmt.Printf("rk: %v\n", rk)
		// fmt.Printf("xk: %v\n", xk)
		// fmt.Printf("gradNorm: %v\n", gradNorm)
		ComputeDoglegStep(xk, df, B, Binv, rk, dx)
		// fmt.Printf("dx from dogleg: %v\n", dx)

		// compute pk
		f0 := f(xk)
		blas.GEMV(1.0, B, dx, 0.0, t1)
		fa0 := f0 // TODO: check this is correct??
		fa1 := f0 + blas.DOT(df, dx) + 0.5*blas.DOT(dx, t1)
		blas.COPY(xk, t1)
		blas.AXPY(1.0, dx, t1)
		f1 := f(t1)
		pk := (f0 - f1) / (fa0 - fa1)

		// fmt.Printf("pk = (%f - %f) / (%f - %f) = %v\n", f0, f1, fa0, fa1, pk)
		// fmt.Printf("rk = %v\n", rk)

		// set values
		if pk < 0.25 {
			// fmt.Println("reducing region; rk = 0.25 * rk")
			rk = 0.25 * rk
		} else {
			if pk > 0.75 && (blas.NRM2(dx)-rk > TOLERANCE) {
				newrk := math.Min(2*rk, RMAX)
				// fmt.Printf("increasing region; rk = %v\n", newrk)
				rk = newrk
			}
		}
		// fmt.Printf("rk* = %v\n", rk)
		if pk > BETA {
			blas.AXPY(1.0, dx, xk)
			// fmt.Printf("accepting step; new xk = %v\n", xk)
		}
		iter++
		gradNorm = gradF(xk, f, df)
	}
	if iter >= maxiter {
		panic("TRUST REGION EXCEEDED MAX ITERATIONS")
	}
	// fmt.Println("--- FINISHED TRUST REGION ---")
	// fmt.Printf("ending gradNorm: %v\n", gradNorm)
	blas.COPY(xk, x1)
}

func ComputeDoglegStep(x0 linalg.Vector, dF linalg.Vector, B linalg.Matrix, Binv linalg.Matrix, rk float64, dx linalg.Vector) {
	dxn := linalg.NewVector(x0.Len())
	dxc := linalg.NewVector(x0.Len())

	// try Newton step
	blas.GEMV(-1.0, Binv, dF, 0.0, dxn)
	crit := blas.NRM2(dxn)
	if crit <= rk {
		// fmt.Printf("dxn: %v\n", dxn)
		// fmt.Printf("|dxn|: %v\n", crit)
		blas.COPY(dxn, dx)
		return
	}

	// try Cauchy step
	blas.GEMV(1.0, B, dF, 0.0, dxc) // dxc = B * dF
	dfBdf := blas.DOT(dF, dxc)      // dfBdf = dF^T * B * dF
	if dfBdf > 0.0 {
		blas.CPSC(-blas.DOT(dF, dF)/dfBdf, dF, dxc)
	} else {
		blas.CPSC(-rk/blas.NRM2(dF), dF, dxc)
	}
	crit = blas.NRM2(dxc)
	if crit >= rk {
		// fmt.Printf("dxc: %v\n", dxc)
		// fmt.Printf("|dxc|: %v\n", crit)
		blas.CPSC(rk/crit, dxc, dx)
		return
	}

	// compute dogleg step
	pb := dxn
	pu := dxc
	a := blas.DOT(pb, pb) - 2*blas.DOT(pb, pu) + blas.DOT(pu, pu)
	b := 2 * (blas.DOT(pb, pu) - blas.DOT(pu, pu))
	c := blas.DOT(pu, pu) - rk*rk

	disc := b*b + a*(rk*rk-c)
	// disc should always be >= 0, take positive root
	eta := (-b + math.Sqrt(disc)) / (2 * a)
	// clamp eta to [0, 2]
	if eta < 0.0 {
		eta = 0.0
	} else if eta > 2.0 {
		eta = 2.0
	}

	if eta <= 1.0 {
		// 0 <= eta <= 1
		blas.CPSC(eta, pu, dx)
	} else {
		// 1 <= eta <= 2
		blas.CPSC(eta-1.0, pb, dx)
		blas.AXPY(eta-1.0, pu, dx)
		blas.AXPY(1.0, pu, dx)
	}

	// fmt.Printf("dx interp: %v\n", dx)
}
