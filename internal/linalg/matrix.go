package linalg

// Matrix represents a dense matrix of floats.
type Matrix struct {
	rows int
	cols int
	data []float64
}

func NewDenseMatrix(rows, cols int) Matrix {
	return Matrix{
		rows: rows,
		cols: cols,
		data: make([]float64, rows*cols),
	}
}

// Get returns the value at the given row and column.
//
// i is the row index, starting at 0.
// j is the column index, starting at 0.
//
// Returns the value at the given row and column.
func (m Matrix) Get(i, j int) float64 {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(ErrIndexOutOfBounds)
	}
	return m.data[i*m.cols+j]
}

func (m Matrix) Set(i, j int, value float64) {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(ErrIndexOutOfBounds)
	}
	m.data[i*m.cols+j] = value
}

func (m Matrix) Rows() int {
	return m.rows
}

func (m Matrix) Cols() int {
	return m.cols
}

func (m Matrix) Identity() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if i == j {
				m.Set(i, j, 1)
			} else {
				m.Set(i, j, 0)
			}
		}
	}
}

func (m Matrix) Copy(n Matrix) {
	if m.rows != n.rows || m.cols != n.cols {
		panic(ErrDimensionMismatch)
	}
	copy(m.data, n.data)
}

func (m Matrix) GetRow(i int, v Vector) {
	if i < 0 || i >= m.rows {
		panic(ErrIndexOutOfBounds)
	}
	for j := 0; j < m.cols; j++ {
		v[j] = m.Get(i, j)
	}
}

func (m Matrix) GetCol(j int, v Vector) {
	if j < 0 || j >= m.cols {
		panic(ErrIndexOutOfBounds)
	}
	start := j * m.rows
	end := start + m.rows
	copy(v, m.data[start:end])
}

func (m Matrix) AddOuterProduct(x Vector, y Vector, alpha float64) {
	if m.rows != x.Len() || m.cols != y.Len() {
		panic(ErrDimensionMismatch)
	}

	for i := range x {
		for j := range y {
			v := m.Get(i, j)
			vv := v + alpha*x[i]*y[j]
			m.Set(i, j, vv)
		}
	}
}

func (m Matrix) GetOuterProduct(x Vector, y Vector, alpha float64) {
	if m.rows != x.Len() || m.cols != y.Len() {
		panic(ErrDimensionMismatch)
	}
	for i := range x {
		for j := range y {
			m.Set(i, j, alpha*x[i]*y[j])
		}
	}
}
