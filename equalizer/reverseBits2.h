#pragma once
#include <stdint.h>

// Utility class to reverse lowest N bits in an integer.
struct ReverseBits
{
	using pfn = uint32_t( *)( uint32_t x );
	const pfn m_pfn;

	// Construct from the N
	ReverseBits( uint32_t count );

	// Reverse these bits
	uint32_t operator()( uint32_t x ) const
	{
		return m_pfn( x );
	}
};