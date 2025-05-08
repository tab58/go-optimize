package optim

import (
	"github.com/tab58/go-optimize/internal/blas"
	"github.com/tab58/go-optimize/internal/linalg"
)

func ParabolicLineSearch(x0, s, x linalg.Vector, f func(x linalg.Vector) float64) float64 {
	if x0.Len() != s.Len() || x0.Len() != x.Len() {
		panic(linalg.ErrDimensionMismatch)
	}
	blas.COPY(x0, x)
	alphas := []float64{0.0, 0.0, 0.0}
	fs := []float64{0.0, 0.0, 0.0}

	// fmt.Printf("searchDir: %v\n", s)

	// calculate starting values
	fs[0] = f(x)
	alphas[0] = 0.0
	alpha := 0.01
	blas.COPY(x0, x)
	blas.AXPY(alpha, s, x)

	fs[1] = f(x)
	alphas[1] = alpha
	alpha = 2 * alpha
	blas.COPY(x0, x)
	blas.AXPY(alpha, s, x)

	fs[2] = f(x)
	alphas[2] = alpha

	// fmt.Printf("fs: %v\n", fs)
	// fmt.Printf("alphas: %v\n", alphas)

	// bracket the function minimum
	j := 2
	for fs[(j-1)%3]-fs[j%3] > 0.0 {
		j = j + 1
		alpha = 2 * alpha
		blas.COPY(x0, x)
		blas.AXPY(alpha, s, x)
		fs[j%3] = f(x)
		alphas[j%3] = alpha

		// fmt.Printf("x: %v\n", x)
		// fmt.Printf("f(x): %v\n", fs[j%3])
		// fmt.Printf("alpha: %v\n", alphas[j%3])
	}
	da := (alphas[j%3] - alphas[(j-1)%3]) / 2
	aLast := alpha - da
	blas.COPY(x0, x)
	blas.AXPY(aLast, s, x)
	fLast := f(x)

	// fmt.Printf("aLast: %f\n", aLast)
	// fmt.Printf("fLast: %f\n", fLast)
	// fmt.Printf("alphas: %v\n", alphas)
	// fmt.Printf("fs: %v\n", fs)

	// fit the parabola
	a2 := 0.0
	f1 := 0.0
	f2 := 0.0
	f3 := 0.0
	if fs[(j-1)%3] < fLast {
		a2 = alphas[(j-1)%3]
		f1 = fs[(j-2)%3]
		f2 = fs[(j-1)%3]
		f3 = fLast
	} else {
		a2 = aLast
		f1 = fs[(j-1)%3]
		f2 = fLast
		f3 = fs[j%3]
	}

	// points now bracket the minimum
	// use the parabolic formula to get the minimum
	aMin := a2 + ((da * (f1 - f3)) / (2 * (f1 - 2*f2 + f3)))
	blas.COPY(x0, x)
	blas.AXPY(aMin, s, x)

	// fmt.Printf("a*: %f + ((%f * (%f - %f)) / (2 * (%f - 2*%f + %f))) = %f\n", a2, da, f1, f3, f1, f2, f3, aMin)

	return f(x)
}
