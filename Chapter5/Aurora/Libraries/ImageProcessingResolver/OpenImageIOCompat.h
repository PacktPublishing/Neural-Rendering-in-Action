// OpenImageIO compatibility header
// Provides minimal std::byte declaration for OpenImageIO headers

#if !defined(__GNUC__) && (!defined(_HAS_STD_BYTE) || (_HAS_STD_BYTE == 0))

// Minimal std::byte declaration for OpenImageIO compatibility
namespace std {
    enum class byte : unsigned char {};
}

#endif
