#pragma once
#include <emmintrin.h>
#include <pmmintrin.h>	// MOVDDUP and ADDSUBPS are from SSE3 set
#include "complex.h"

// ==== Miscellaneous utilities ====
using XMVECTOR2 = __m128;
using XMVECTOR = __m128;

// Load 2D vector from memory
__forceinline XMVECTOR2 loadFloat2( const void* pointer )
{
	return _mm_castpd_ps( _mm_load_sd( (const double*)pointer ) );
}

// [ a, b, a, b ] where [ a, b ] are 2 floats at the address
__forceinline XMVECTOR loadFloat2Dup( const void* pointer )
{
	// Duplicating while loading is free, compare these two instructions:
	// https://uops.info/html-instr/MOVDDUP_XMM_M64.html
	// https://uops.info/html-instr/MOVSD_XMM_XMM_M64.html
	return _mm_castpd_ps( _mm_loaddup_pd( (const double*)pointer ) );
}

// Store a 2D vector
__forceinline void storeFloat2( void* pointer, XMVECTOR2 vec )
{
	_mm_store_sd( (double*)pointer, _mm_castps_pd( vec ) );
}

// [ a, b, c, d ] => [ c, d, a, b ]
__forceinline XMVECTOR flipHighLow( XMVECTOR v )
{
	return _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) );
}

__forceinline XMVECTOR2 getHigh( XMVECTOR v )
{
	return _mm_movehl_ps( v, v );
}

// ==== Vectorizing fft_init ====

// pow( M_E, complex{ 0, valScalar } )
// https://en.wikipedia.org/wiki/Complex_number#Exponential_function, Euler's formula says the result is
// [ cos( val ), sin( val ) ]
XMVECTOR computeOmegaVec( const float valScalar );

// Same as above, for 2 different angles at once
XMVECTOR computeOmegaVec_x2( const XMVECTOR2 angles );

// Same as above, for 4 different angles at once
std::pair<XMVECTOR, XMVECTOR> computeOmegaVec_x4( const XMVECTOR angles );

// ==== Vectorizing fft_run ====

// Multiply 2 complex numbers, the result vector is duplicated, [ a, b, a, b ] where a = real, b = imaginary
// The x input value must be duplicated the same way, for y the higher 2 lanes are ignored
__forceinline XMVECTOR multiplyComplex( XMVECTOR2 x, XMVECTOR2 y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]
	// BTW, it takes only 1 instruction less compared to scalar code, 5 versus 6.
	// The win comes from the fact that only 2 of them have >1 cycle latency, shuffles are really fast.

	// x is [ a, b, a, b ], duplicated while loading from memory
	y = _mm_unpacklo_ps( y, y );			// [ c, c, d, d ]
	const __m128 prod = _mm_mul_ps( x, y );	// [ ac, bc, ad, bd ]

	// Modern CPUs can run 2 of these per clock, just what we need as there's no data dependency
	const __m128 r1 = _mm_shuffle_ps( prod, prod, _MM_SHUFFLE( 2, 0, 2, 0 ) );	// [ ac, ad, ac, ad ]
	const __m128 r2 = _mm_shuffle_ps( prod, prod, _MM_SHUFFLE( 1, 3, 1, 3 ) );	// [ bd, bc, bd, bc ]

	// Pretty sure someone at Intel was thinking about complex numbers in early 2000s when they added addsubps to P4 Prescott
	const __m128 res = _mm_addsub_ps( r1, r2 );	// [ ac - bd, ad + bc, ac - bd, ad + bc ]
	return res;
}

// a1 = a1 + om * a2; a2 = a1 - om * a2
__forceinline void fftMainLoop( const complex& om, complex& a1c, complex& a2c )
{
	const XMVECTOR2 omega = loadFloat2Dup( &om );
	const XMVECTOR2 a1 = loadFloat2( &a2c );
	const XMVECTOR a0dup = loadFloat2Dup( &a1c );

	XMVECTOR product = multiplyComplex( omega, a1 );
	product = _mm_xor_ps( product, _mm_setr_ps( 0, 0, -0.0f, -0.0f ) );
	const XMVECTOR result = _mm_add_ps( a0dup, product );
	storeFloat2( &a1c, result );
	storeFloat2( &a2c, getHigh( result ) );
}