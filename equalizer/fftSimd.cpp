#pragma once
#include "stdafx.h"
#include <smmintrin.h>	// SSE 4.1
#include <immintrin.h>	// FMA
#include "fftSimd.h"

template<int destLane, int sourceLane, int zeroLanes>
constexpr int insertMask()
{
	static_assert( destLane >= 0 && destLane < 4 );
	static_assert( sourceLane >= 0 && sourceLane < 4 );
	static_assert( zeroLanes >= 0 && zeroLanes < 16 );
	return ( sourceLane << 6 ) | ( destLane << 4 ) | zeroLanes;
}

// Extract a scalar, move to designated destination lane, zero out the rest of the lanes.
// Sounds complicated but the instruction has 1 cycle latency, and runs 2 per clock on modern AMD CPUs.
template<int sourceLane, int destLane = 0>
__forceinline __m128 extractSingle( __m128 src )
{
	constexpr int zeroLanes = 0b1111 ^ ( 1 << destLane );
	constexpr int mask = insertMask<destLane, sourceLane, zeroLanes>();
	return _mm_insert_ps( src, src, mask );
}

// [ a, b, c, d ] => [ c, d, c, d ]
__forceinline __m128 upper( __m128 v )
{
	return _mm_movehl_ps( v, v );
}

// Aggregate magic numbers to save registers
constexpr float XM_1DIV2PI = 0.159154943f;
constexpr float XM_2PI = 6.283185307f;
constexpr float XM_PIDIV2 = 1.570796327f;
constexpr float XM_PI = 3.141592654f;
// [ 1 / ( 2 * pi ), 2 * pi, pi / 2, pi ]
static const __m128 s_piConstants = _mm_setr_ps( XM_1DIV2PI, XM_2PI, XM_PIDIV2, XM_PI );

// [ sign bit, +1, -1, 0 ]
static const __m128 s_miscConstants = _mm_setr_ps( -0.0f, +1, -1, 0 );

// DirectXMath code:
// 11-degree minimax approximation
// *pSin = ( ( ( ( ( -2.3889859e-08f * y2 + 2.7525562e-06f ) * y2 - 0.00019840874f ) * y2 + 0.0083333310f ) * y2 - 0.16666667f ) * y2 + 1.0f ) * y;
// 10-degree minimax approximation
// const float p = ( ( ( ( -2.6051615e-07f * y2 + 2.4760495e-05f ) * y2 - 0.0013888378f ) * y2 + 0.041666638f ) * y2 - 0.5f ) * y2 + 1.0f;
// *pCos = sign * p;

// We want the cosines in the even lanes of the output, the coefficients are interleaved from lower to upper.

static const __m128 s_sinCosCoeffs0 = _mm_setr_ps( -2.6051615e-07f, -2.3889859e-08f, 2.4760495e-05f, 2.7525562e-06f );
static const __m128 s_sinCosCoeffs1 = _mm_setr_ps( -0.0013888378f, -0.00019840874f, 0.041666638f, 0.0083333310f );
static const __m128 s_sinCosCoeffs2 = _mm_setr_ps( -0.5f, -0.16666667f, 1, 1 );

XMVECTOR computeOmegaVec( const float valScalar )
{
	// [ 1 / ( 2 * pi ), 2 * pi, pi / 2, pi ]
	const __m128 piConstants = s_piConstants;

	__m128 val = _mm_setr_ps( valScalar, 0, 0, 0 );

	// quotient = round( val / ( 2 * Pi ) )
	const __m128 quotient = _mm_round_ps( _mm_mul_ss( val, piConstants ), _MM_FROUND_NINT );

	// val -= XM_2PI * quotient; because we rounded to the nearest integer, this makes the angle into [-pi .. +pi] interval
	__m128 y = _mm_fnmadd_ss( quotient, _mm_movehdup_ps( piConstants ), val );

	// Map y to [-pi/2,pi/2] with sin(y) = sin(Value)

	// [ sign bit, +1, -1, 0 ]
	const __m128 miscConstants = s_miscConstants;
	const __m128 yAbs = _mm_andnot_ps( miscConstants, y );
	__m128 finalMultiplier;
	if( _mm_comigt_ss( yAbs, extractSingle<2>( piConstants ) ) )
	{
		// y > XM_PIDIV2 or y < -XM_PIDIV2

		const __m128 sign = _mm_and_ps( miscConstants, y );
		__m128 pi = extractSingle<3>( piConstants );
		pi = _mm_or_ps( pi, sign );	// That thing equals to +pi when ( y > XM_PIDIV2 ), or -pi when ( y < -XM_PIDIV2 )
		y = _mm_sub_ss( pi, y );

		// sign = -1
		// We need this in the last multiplier: [ -1, y, 0, 0 ]
		finalMultiplier = _mm_moveldup_ps( y );
		constexpr int mask = insertMask<0, 2, 0b1100>();
		finalMultiplier = _mm_insert_ps( finalMultiplier, miscConstants, mask );
	}
	else
	{
		// sign = +1
		// We need this in the last multiplier: [ +1, y, 0, 0 ]
		finalMultiplier = _mm_moveldup_ps( y );
		constexpr int mask = insertMask<0, 1, 0b1100>();
		finalMultiplier = _mm_insert_ps( finalMultiplier, miscConstants, mask );
	}
	const __m128 y2 = _mm_moveldup_ps( _mm_mul_ss( y, y ) );

	// And the main formula now only takes 10 instructions.
	__m128 coeffs = s_sinCosCoeffs0;
	__m128 v = _mm_fmadd_ps( y2, coeffs, upper( coeffs ) );

	coeffs = s_sinCosCoeffs1;
	v = _mm_fmadd_ps( y2, v, coeffs );
	v = _mm_fmadd_ps( y2, v, upper( coeffs ) );

	coeffs = s_sinCosCoeffs2;
	v = _mm_fmadd_ps( y2, v, coeffs );
	v = _mm_fmadd_ps( y2, v, upper( coeffs ) );

	return _mm_mul_ps( v, finalMultiplier );
}

template<int i>
__forceinline __m128 broadcastLane( __m128 x )
{
	static_assert( i >= 0 && i < 4 );
	return _mm_shuffle_ps( x, x, _MM_SHUFFLE( i, i, i, i ) );
}

std::pair<XMVECTOR, XMVECTOR> computeOmegaVec_x4( const XMVECTOR angles )
{
	// [ 1 / ( 2 * pi ), 2 * pi, pi / 2, pi ]
	const __m128 piConstants = s_piConstants;

	// Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
	__m128 val = angles;
	const __m128 quotient = _mm_round_ps( _mm_mul_ps( val, broadcastLane<0>( piConstants ) ), _MM_FROUND_NINT );
	__m128 y = _mm_fnmadd_ps( quotient, broadcastLane<1>( piConstants ), val );

	// [ sign bit, +1, -1, 0 ]
	const __m128 miscConstants = s_miscConstants;

	// Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
	const __m128 signBit = broadcastLane<0>( miscConstants );
	const __m128 yAbs = _mm_andnot_ps( signBit, y );
	const __m128 pidiv2 = broadcastLane<2>( piConstants );
	const __m128 absExceeds90Mask = _mm_cmpgt_ps( yAbs, pidiv2 );

	// We don't want to predict any branches here.
	// Computing both sizes of the "if" and selecting the right one with _mm_blendv_ps.
	// These extra computations is nothing compared to the cost of branch prediction.
	const __m128 sign = _mm_and_ps( signBit, y );
	__m128 pi = broadcastLane<3>( piConstants );
	pi = _mm_or_ps( pi, sign );
	const __m128 case1_y = _mm_sub_ps( pi, y );
	const __m128 case1_cosMul = broadcastLane<2>( miscConstants );	// -1 in all 4 lanes
	const __m128 case2_cosMul = broadcastLane<1>( miscConstants );	// +1 in all 4 lanes
	const __m128 cosMul = _mm_blendv_ps( case2_cosMul, case1_cosMul, absExceeds90Mask );
	y = _mm_blendv_ps( y, case1_y, absExceeds90Mask );

	// Prepare two vectors with final multipliers.
	// These are 4 pairs of values, each being [ +/-1, y ]
	const __m128 finalMultiplierLow = _mm_unpacklo_ps( cosMul, y );
	const __m128 finalMultiplierHigh = _mm_unpackhi_ps( cosMul, y );

	// Prepare two vectors with y^2, one with [ y0^2, y0^2, y1^2, y1^2 ], another one with [ y2^2, y2^2, y3^2, y3^2 ]
	const __m128 y2_all = _mm_mul_ps( y, y );
	const __m128 y2low = _mm_unpacklo_ps( y2_all, y2_all );
	const __m128 y2high = _mm_unpackhi_ps( y2_all, y2_all );

	// Now use FMA to compute the result.
	// Unlike the code in computeOmegaVec, we now have 2 independent streams of data, 
	// Should help saturating the EUs despite 4-5 cycles of latency of every FMA instruction.
	__m128 coeffs = s_sinCosCoeffs0;
	const __m128 coeffs0_1 = _mm_movelh_ps( coeffs, coeffs );
	const __m128 coeffs0_2 = _mm_movehl_ps( coeffs, coeffs );
	__m128 low = _mm_fmadd_ps( y2low, coeffs0_1, coeffs0_2 );
	__m128 high = _mm_fmadd_ps( y2high, coeffs0_1, coeffs0_2 );

	coeffs = s_sinCosCoeffs1;
	__m128 tmp = _mm_movelh_ps( coeffs, coeffs );
	low = _mm_fmadd_ps( y2low, low, tmp );
	high = _mm_fmadd_ps( y2high, high, tmp );

	tmp = _mm_movehl_ps( coeffs, coeffs );
	low = _mm_fmadd_ps( y2low, low, tmp );
	high = _mm_fmadd_ps( y2high, high, tmp );

	coeffs = s_sinCosCoeffs2;
	tmp = _mm_movelh_ps( coeffs, coeffs );
	low = _mm_fmadd_ps( y2low, low, tmp );
	high = _mm_fmadd_ps( y2high, high, tmp );

	tmp = _mm_movehl_ps( coeffs, coeffs );
	low = _mm_fmadd_ps( y2low, low, tmp );
	high = _mm_fmadd_ps( y2high, high, tmp );

	low = _mm_mul_ps( low, finalMultiplierLow );
	high = _mm_mul_ps( high, finalMultiplierHigh );

	return std::make_pair( low, high );
}