#ifndef USE_REFERENCE_BSDF
#define USE_REFERENCE_BSDF 0
#endif

#if USE_REFERENCE_BSDF
#include "ReferenceBSDF.slang"
#else
#include "StandardSurfaceBSDF.slang"
#endif
