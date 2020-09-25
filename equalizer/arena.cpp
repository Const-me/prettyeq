#include "stdafx.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "arena.h"
#ifdef WIN32
static inline void* allocateMemory( size_t bytes )
{
	// Trying to optimize this with better calls is useless, IMO.
	// Modern allocators have a size threshold where they stop fooling around with user mode heaps.
	return HeapAlloc( GetProcessHeap(), 0, bytes );
}

static inline void freeMemory( void*& pointer, size_t map_size )
{
	HeapFree( GetProcessHeap(), 0, pointer );
	pointer = nullptr;
}
#else
static inline void* allocateMemory( size_t map_size )
{
	void *mem = mmap(
		nullptr,
		map_size,
		PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
		-1, 0 );

	if( mem == MAP_FAILED )
		return nullptr;
	return mem;
}

static inline void freeMemory( void*& pointer, size_t map_size )
{
	munmap( pointer, map_size );
	pointer = nullptr;
}
#endif

inline chunk_t* CHUNK_OFFSET( void* memory )
{
	return (chunk_t*)( ( (uint8_t*)memory ) - sizeof( chunk_t ) );
}

inline void* MEMORY_OFFSET( chunk_t *chunk )
{
	return ( (uint8_t*)chunk ) + sizeof( chunk_t );
}

inline chunk_t* NEXT_CHUNK( chunk_t *chunk, size_t size )
{
	uint8_t* pb = (uint8_t*)MEMORY_OFFSET( chunk );
	pb += size;
	return (chunk_t*)pb;
}

arena_t* arena_new( size_t num_chunks, size_t chunk_size )
{
	assert( num_chunks > 0 );
	assert( chunk_size > 0 );

	arena_t *arena = (arena_t *)calloc( sizeof( arena_t ), 1 );
	if( !arena )
		return nullptr;

	size_t map_size = num_chunks * ( chunk_size + sizeof( chunk_t ) );
	void * const mem = allocateMemory( map_size );
	if( nullptr == mem )
		return nullptr;

	chunk_t *chunk = (chunk_t *)mem;
	for( unsigned int i = 0; i < num_chunks - 1; i++ )
	{
		chunk->next = NEXT_CHUNK( chunk, chunk_size );
		chunk = chunk->next;
	}
	chunk->next = nullptr;

	arena->chunk_size = chunk_size;
	arena->map_size = map_size;
	arena->mem = mem;
	arena->avail_chunk = (chunk_t *)mem;

	return arena;
}

void* arena_alloc( arena_t *arena )
{
	assert( arena );
	assert( arena->avail_chunk );

	void *p = MEMORY_OFFSET( arena->avail_chunk );
	assert( p );
	arena->avail_chunk = arena->avail_chunk->next;
	return p;
}

void arena_dealloc( arena_t *arena, void *mem ) {
	assert( arena );
	assert( mem );

	chunk_t *chunk = CHUNK_OFFSET( mem );
	chunk->next = arena->avail_chunk;
	arena->avail_chunk = chunk;
}

void arena_destroy( arena_t **arena )
{
	assert( arena );
	assert( ( *arena )->mem );
	freeMemory( ( *arena )->mem, ( *arena )->map_size );
	free( *arena );
	*arena = nullptr;
}