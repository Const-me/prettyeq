#pragma once
#include <stdint.h>

// Utility class to reverse lowest N bits in an integer.
class ReverseBits
{
	// CPUs have indirect branch predictor, gonna help because the client code gonna `call rbx` over and over again with the same address.
	using pfn = uint32_t( *)( uint32_t x );
	const pfn m_pfn;

	// We use C++ function template to generate 33 separate implementations, all of them are branchless and contain minimum count of instructions, and therefore latency.
	// If you're porting that to C, you can write a script in Python or any other language to generate the source code, for all 33 of them.
	static const std::array<pfn, 33> s_implementations;

public:

	// Construct from the N
	ReverseBits( uint32_t count ) :
		m_pfn( s_implementations[ count ] )
	{
		assert( count <= 32 );
	}

	// Reverse these bits
	uint32_t operator()( uint32_t x ) const
	{
		return m_pfn( x );
	}
};