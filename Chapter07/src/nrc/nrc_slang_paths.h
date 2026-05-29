// Slang module search paths, populated at CMake-configure time.
//
// The host's Slang session needs to resolve imports such as `import MLP;`
// (whose primary module lives in RTXNS). Rather than bake those paths into
// shader sources or environment variables, we expose them through this
// accessor, generated via configure_file().

#pragma once

#include <cstddef>

namespace nrc {

// Returns a pointer to an array of UTF-8 path strings. *count is set to the
// number of entries. Pointer lifetime is program-lifetime (static storage).
const char* const* slangIncludePaths(size_t* count);

}  // namespace nrc
