#pragma once
#include <stdint.h>
#include <emmintrin.h>

// Utility class to reverse lowest N bits in an integer.
struct ReverseBits
{
	const __m128i lookupTable;
	using pfn = uint32_t( __vectorcall * )( uint32_t x, __m128i table );
	const pfn m_pfn;

	// Construct from the N
	ReverseBits( uint32_t count );

	// Reverse these bits
	uint32_t operator()( uint32_t x ) const
	{
		return m_pfn( x, lookupTable );
	}
};