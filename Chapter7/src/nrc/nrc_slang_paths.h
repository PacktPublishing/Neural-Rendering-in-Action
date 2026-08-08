// Slang shader include paths, populated at CMake-configure time.
//
// The host's Slang session needs to be able to resolve #include directives
// like `#include "MLP.slang"` (which lives in RTXNS). Rather than bake those
// paths into shader sources or environment variables, we expose them through
// this accessor, generated via configure_file().

#pragma once

#include <cstddef>

namespace nrc {

// Returns a pointer to an array of UTF-8 path strings. *count is set to the
// number of entries. Pointer lifetime is program-lifetime (static storage).
const char* const* slangIncludePaths(size_t* count);

}  // namespace nrc
