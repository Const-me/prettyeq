#pragma once
// The problem with std::complex - it has a non-trivial constructor.
// Profiler shows like 20% of time was wasted constructing elements in that static lookup table.
struct complex
{
	float real, imaginary;

	complex() = default;
	complex( float f ) : real( f ), imaginary( 0 ) { }
	complex( float r, float i ) : real( r ), imaginary( i ) { }

	// Only used for asserts in debug builds
	bool operator==( const complex& that ) const
	{
		const __m128 a = loadFloat2( this );
		const __m128 b = loadFloat2( &that );
		return 0 == _mm_movemask_ps( _mm_cmpneq_ps( a, b ) );
	}

	// Only used for asserts in debug builds
	bool closeEnough( float r, float i, float tolerance = 0.000001f ) const
	{
		assert( tolerance > 0 );
		const __m128 a = loadFloat2( this );
		const __m128 b = _mm_setr_ps( r, i, 0, 0 );
		const __m128 diff = _mm_sub_ps( a, b );
		const __m128 absDiff = _mm_andnot_ps( _mm_set1_ps( -0.0f ), diff );
		return 0 == _mm_movemask_ps( _mm_cmpgt_ps( absDiff, _mm_set1_ps( tolerance ) ) );
	}
};