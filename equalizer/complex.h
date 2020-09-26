#pragma once
#if 1
#define STD_COMPLEX 1
#include <complex>
using complex = std::complex<float>;
#else
#define STD_COMPLEX 0
// The problem with std::complex - it has a non-trivial constructor.
// Profiler shows like 20% of time was wasted constructing elements in that static lookup table.
struct complex
{
	float real, imaginary;

	complex() = default;
	complex( float f ) : real( f ), imaginary( 0 ) { }
	complex( float r, float i ) : real( r ), imaginary( i ) { }
};
#endif