#pragma once
#include <chrono>

// Utility class to measure time
class Stopwatch
{
	using clock = std::chrono::high_resolution_clock;
	clock::time_point begin = clock::now();

	template<class Unit>
	double elapsed() const
	{
		using namespace std::chrono;
		using us = duration<double, Unit>;
		const us elapsed = duration_cast<us>( clock::now() - begin );
		return elapsed.count();
	}

public:
	double elapsedMicroseconds() const
	{
		return elapsed<std::micro>();
	}
	double elapsedMilliseconds() const
	{
		return elapsed<std::milli>();
	}
	double elapsedSeconds() const
	{
		return elapsed<std::ratio<1, 1>>();
	}
};