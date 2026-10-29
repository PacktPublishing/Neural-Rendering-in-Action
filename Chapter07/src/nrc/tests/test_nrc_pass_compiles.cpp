// Compile-only smoke test that nrc_pass.h and nrc_pass_integration.h can be
// included and the default-constructed NrcPass works. We do not call init()
// because that requires a real Vulkan device.

#include "nrc/nrc_pass.h"
#include "nrc/nrc_pass_integration.h"

int main() {
    nrc::NrcPass pass;  // ensure default ctor compiles
    (void)pass;

    // Entry points reachable; do not call with null handles.
    (void)&nrc_pass_init;
    (void)&nrc_pass_destroy;
    (void)&nrc_pass_record_frame;
    (void)&nrc_pass_descriptor_set;
    (void)&nrc_pass_descriptor_set_layout;
    return 0;
}
