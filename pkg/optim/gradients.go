package optim

import (
	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
)

type gradientFunc func(x linalg.Vector, f ObjectiveFunc, gradF linalg.Vector) float64

func ForwardGradientConstantStep(delta float64, n int) gradientFunc {
	dx := linalg.NewVector(n)
	dx.Set(delta)
	return func(x linalg.Vector, f ObjectiveFunc, gradF linalg.Vector) float64 {
		xi := 0.0
		dxi := 0.0
		g := 0.0
		nrm2 := 0.0
		fxh := 0.0
		fx0 := f(x)

		for i := range x {
			xi = x[i]
			dxi = dx[i]

			// calculate the gradient
			x[i] = xi + dxi
			fxh = f(x)
			g = (fxh - fx0) / dxi

			// restore the original value
			x[i] = xi

			// calculate the norm of the gradient
			nrm2 = blas.Hypot(nrm2, g)
		}
		return nrm2
	}
}

func BackwardGradientConstantStep(delta float64, n int) gradientFunc {
	dx := linalg.NewVector(n)
	dx.Set(delta)
	return func(x linalg.Vector, f ObjectiveFunc, gradF linalg.Vector) float64 {
		xi := 0.0
		dxi := 0.0
		g := 0.0
		nrm2 := 0.0
		fx0 := 0.0
		fxh := f(x)

		for i := range x {
			xi = x[i]
			dxi = dx[i]

			// calculate the gradient
			x[i] = xi - dxi
			fx0 = f(x)
			g = (fxh - fx0) / dxi

			// restore the original value
			x[i] = xi

			// calculate the norm of the gradient
			nrm2 = blas.Hypot(nrm2, g)
		}
		return nrm2
	}
}

func CentralGradientConstantStep(delta float64, n int) gradientFunc {
	dx := linalg.NewVector(n)
	dx.Set(delta)
	return func(x linalg.Vector, f ObjectiveFunc, gradF linalg.Vector) float64 {
		xi := 0.0
		dxi := 0.0
		g := 0.0
		nrm2 := 0.0
		fx0 := 0.0
		fx1 := 0.0

		for i := range x {
			xi = x[i]
			dxi = dx[i]

			// calculate the gradient
			x[i] = xi + dxi
			fx1 = f(x)
			x[i] = xi - dxi
			fx0 = f(x)
			g = (fx1 - fx0) / (2 * dxi)
			gradF[i] = g

			// restore the original value
			x[i] = xi

			// calculate the norm of the gradient
			nrm2 = blas.Hypot(nrm2, g)
		}
		return nrm2
	}
}
