package optim

import (
	"math"

	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
)

type ObjectiveFunc func(x linalg.Vector) float64

// solveOptions are the options for an unconstrained optimization.
type solveOptions struct {
	tolerance     float64
	maxIterations int
	gradientFunc  gradientFunc
}

func WithTolerance(tolerance float64) func(*solveOptions) {
	return func(opts *solveOptions) {
		opts.tolerance = tolerance
	}
}

func WithMaxIterations(maxIterations int) func(*solveOptions) {
	return func(opts *solveOptions) {
		opts.maxIterations = maxIterations
	}
}

func WithGradientFunc(gradientFunc gradientFunc) func(*solveOptions) {
	return func(opts *solveOptions) {
		opts.gradientFunc = gradientFunc
	}
}

// solution is the result of an unconstrained optimization.
type solution struct {
	ValidSolution bool
	Result        linalg.Vector
	Iterations    int
	GradientNorm  float64
	Objective     float64
}

type quasiNewtonSolver struct {
	rankUpdateFunc func(N linalg.Matrix, y linalg.Vector, dx linalg.Vector)
}

func (s *quasiNewtonSolver) Solve(f ObjectiveFunc, x0 linalg.Vector, options ...func(*solveOptions)) *solution {
	opts := &solveOptions{
		tolerance:     1e-8,
		maxIterations: 1000,
		gradientFunc:  CentralGradientConstantStep(1e-4, x0.Len()),
	}

	for _, option := range options {
		option(opts)
	}

	return s.solve(f, x0, opts)
}

func (s *quasiNewtonSolver) solve(f ObjectiveFunc, x0 linalg.Vector, opts *solveOptions) *solution {
	evaluateGradient := opts.gradientFunc
	updateHessianInverse := s.rankUpdateFunc
	tolerance := opts.tolerance
	maxIterations := opts.maxIterations

	n := x0.Len()
	x1 := linalg.NewVector(n)
	dx := linalg.NewVector(n)
	g0 := linalg.NewVector(n)
	g1 := linalg.NewVector(n)
	y := linalg.NewVector(n)
	B := linalg.NewDenseMatrix(n, n)
	Binv := linalg.NewDenseMatrix(n, n)

	f0 := math.Inf(1)
	f1 := f(x0)
	B.Identity()
	Binv.Identity()

	// fmt.Printf("x0: %v\n", x0)
	// fmt.Printf("f0: %v\n", f0)

	gradNorm := evaluateGradient(x0, f, g0)
	blas.CPSC(-1.0/gradNorm, g0, y) // search direction is s = -grad(f)

	// fmt.Printf("g0: %v\n", g0)
	// fmt.Printf("|g0|: %v\n", gradNorm)

	iter := 0
	var temp1 linalg.Vector
	var temp2 linalg.Vector
	for math.Abs(f1-f0) > tolerance && math.Abs(gradNorm) > tolerance && iter < maxIterations {
		// fmt.Printf("--- ITERATION %d ---\n", iter)
		// ParabolicLineSearch(x0, y, x1, f)
		// blas.AXPY(1.0, x0, x1)
		ComputeTrustRegion(x0, f, evaluateGradient, x1, B, Binv)
		gradNorm = evaluateGradient(x1, f, g1)
		f0 = f1
		f1 = f(x1)

		// fmt.Printf("x%d: %v\n", iter+1, x1)
		// fmt.Printf("g%d: %v\n", iter+1, g1)
		// fmt.Printf("|g%d|: %v\n", iter+1, gradNorm)
		// fmt.Printf("f%d: %v\n", iter+1, f1)

		blas.COPY(x1, dx)       // dx = x1
		blas.AXPY(-1.0, x0, dx) // dx = x1 - x0
		blas.COPY(g1, y)        // y = g1
		blas.AXPY(-1.0, g0, y)  // y = g1 - g0 = -(grad(f))

		// fmt.Printf("dGradF: %v\n", y)
		// fmt.Printf("deltaX: %v\n", dx)

		updateHessianInverse(Binv, y, dx)
		UpdateHessianBFGS(B, y, dx)

		// fmt.Printf("H^-1 approx: %v\n", N)

		blas.GEMV(-1.0/gradNorm, Binv, g1, 0.0, y) // y = -H^-1 * grad(f) / |grad(f)|
		temp1 = g0
		g0 = g1
		g1 = temp1
		temp2 = x0
		x0 = x1
		x1 = temp2

		iter++
	}

	result := linalg.NewVector(n)
	blas.COPY(x1, result)

	return &solution{
		ValidSolution: (math.Abs(f1-f0) <= tolerance) || (math.Abs(gradNorm) <= tolerance),
		Result:        result,
		Iterations:    iter,
		GradientNorm:  gradNorm,
		Objective:     f1,
	}
}

func NewQuasiNewtonSolver() *quasiNewtonSolver {
	return &quasiNewtonSolver{
		rankUpdateFunc: UpdateHessianInverseBFGS,
	}
}
