package optim_test

import (
	"fmt"
	"testing"

	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
	"github.com/tab58/go-optimize/pkg/optim"
)

func SimpleTestFunction(X linalg.Vector) float64 {
	x1 := X[0]
	x2 := X[1]
	return x1*x1 - 2*x1*x2 + 4*x2*x2
}

func SimpleTestFunctionGradient(X linalg.Vector, f optim.ObjectiveFunc, gradF linalg.Vector) float64 {
	x1 := X[0]
	x2 := X[1]
	g1 := 2*x1 - 2*x2
	g2 := -2*x1 + 8*x2
	gradF[0] = g1
	gradF[1] = g2
	return blas.Hypot(g1, g2)
}

func TestQuasiNewtonSolver_AnalyticGradient(t *testing.T) {
	solver := optim.NewQuasiNewtonSolver()

	x0 := linalg.NewVector(2)
	x0[0] = -3
	x0[1] = 1

	TOLERANCE := 1e-11
	MAX_ITERATIONS := 3

	solution := solver.Solve(SimpleTestFunction, x0,
		optim.WithTolerance(TOLERANCE),
		optim.WithMaxIterations(MAX_ITERATIONS),
		optim.WithGradientFunc(SimpleTestFunctionGradient),
	)

	fmt.Printf("solution: %+v\n", solution)

	if !solution.ValidSolution {
		t.Errorf("Expected valid solution, got %v", solution)
	}

	if solution.Iterations > MAX_ITERATIONS {
		t.Errorf("Expected %d iterations, got %d", MAX_ITERATIONS, solution.Iterations)
	}

	if solution.Objective > TOLERANCE {
		t.Errorf("Expected objective to be less than %f, got %f", TOLERANCE, solution.Objective)
	}
}

func TestQuasiNewtonSolver_NumericalGradient(t *testing.T) {
	solver := optim.NewQuasiNewtonSolver()

	x0 := linalg.NewVector(2)
	x0[0] = -3
	x0[1] = 1

	TOLERANCE := 1e-11
	MAX_ITERATIONS := 3

	solution := solver.Solve(SimpleTestFunction, x0,
		optim.WithTolerance(TOLERANCE),
		optim.WithMaxIterations(MAX_ITERATIONS),
	)

	fmt.Printf("solution: %+v\n", solution)

	if !solution.ValidSolution {
		t.Errorf("Expected valid solution, got %v", solution)
	}

	if solution.Iterations > MAX_ITERATIONS {
		t.Errorf("Expected %d iterations, got %d", MAX_ITERATIONS, solution.Iterations)
	}

	if solution.Objective > TOLERANCE {
		t.Errorf("Expected objective to be less than %f, got %f", TOLERANCE, solution.Objective)
	}
}
