package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"math"
	"time"
)

func main() {

	// +++ Started +++
	start := time.Now()

	// +++ System Matrices +++
	A := mat64.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat64.NewDense(2, 1, []float64{0, 1})
	K := mat64.NewDense(1, 2, []float64{-2, -2})
	P := mat64.NewDense(2, 2, []float64{1.2500, 0.2500, 0.2500, 0.3750})

	// +++ System Parameters +++
	m := 1.0
	c := 2.0
	k := 3.0

	// +++ Initial Values +++
	x := mat64.NewDense(2, 1, []float64{0, 5})
	gammaHat := mat64.NewDense(2, 1, []float64{0, 0})

	// +++ Time Constants +++
	dt := 0.0001
	T := 50.0

	// +++ Real System Parameters +++
	gamma := mat64.NewDense(2, 1, []float64{k / m, c / m})

	var lineData = plotter.XYs{}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "State #1 : Position"
	p.X.Label.Text = "time [s]"

	plotSeriesX := make([]float64, int(math.Ceil(T/dt))+1)
	plotSeriesY := make([]float64, int(math.Ceil(T/dt))+1)

	i := 0
	for t := 0.0; t <= T; t = t + dt {

		u := mat64.NewDense(1, 1, make([]float64, 1))
		dotX := mat64.NewDense(2, 1, make([]float64, 2))
		scale := mat64.NewDense(1, 1, make([]float64, 1))
		twoRowOneCol := mat64.NewDense(2, 1, make([]float64, 2))
		dotGammaHat := mat64.NewDense(2, 1, make([]float64, 2))

		// +++ Stabilizing Input +++
		scale.Mul(K, x)
		u.Add(u, scale)

		scale.Mul(gammaHat.T(), x)
		u.Sub(u, scale)
		// --- Stabilizing Input ---

		// +++ State Space Part +++
		twoRowOneCol.Mul(A, x)
		dotX.Add(dotX, twoRowOneCol)

		twoRowOneCol.Product(B, gamma.T(), x)
		dotX.Add(dotX, twoRowOneCol)

		twoRowOneCol.Product(B, u)
		dotX.Add(dotX, twoRowOneCol)
		// --- State Space Part ---

		// +++ Update Law +++
		dotGammaHat.Product(x, B.T(), P, x)
		// --- Update Law ---

		// +++ Iterating The Plant +++
		dotX.Scale(dt, dotX)
		x.Add(x, dotX)
		// --- Iterating The Plant ---

		// +++ Iterating Update Law +++
		dotGammaHat.Scale(dt, dotGammaHat)
		gammaHat.Add(gammaHat, dotGammaHat)
		// --- Iterating Update Law ---

		plotSeriesX[i] = t
		plotSeriesY[i] = x.At(0, 0)
		i++
	}

	// Make a line plotter and set its style.
	lineData = addDataToGraph(plotSeriesX, plotSeriesY)
	l, err := plotter.NewLine(lineData)
	if err != nil {
		panic(err)
	}

	p.Add(l)

	elapsed := time.Since(start)
	fmt.Printf("Elapsed time => %f", elapsed.Seconds())

}

func addDataToGraph(x []float64, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}
	return pts
}
