#pragma once
#include <stdint.h>

// Utility class to reverse lowest N bits in an integer.
struct ReverseBits
{
	// CPUs have indirect branch predictor, gonna help because the client code will `call rbx` over and over again with the same address.
	// We use C++ template to generate 33 separate implementations, all of them are branchless and contain minimum count of instructions, and therefore latency.
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