package linalg

import "errors"

var (
	ErrIndexOutOfBounds = errors.New("index out of bounds")
	ErrDimensionMismatch = errors.New("dimension mismatch")
)
