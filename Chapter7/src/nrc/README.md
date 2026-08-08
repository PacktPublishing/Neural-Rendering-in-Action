# NRC shader architecture

Chapter 7 exposes the NRC implementation as a single Slang module, `nrc`, and
reaches the path tracer through one adapter header.

## Module organization

`nrc.slang` is the only importable NRC module. Its implementation is divided
into cohesive source files with Slang's semantic `__include` mechanism:

```text
nrc.slang                         module nrc
  |-- nrc/records.slang           implementing nrc
  |-- nrc/encoders.slang          implementing nrc
  |-- nrc/features.slang          implementing nrc
  |-- nrc/estimator.slang         implementing nrc
  `-- nrc/network.slang           implementing nrc

gltf_pathtrace.slang
  `-- #include nrc_renderer_adapter.h.slang   renderer adapter

nrc_cache_query.comp.slang
nrc_record_complete.comp.slang
nrc_train.comp.slang
  `-- import nrc
```

`__include` adds implementation files to a Slang module. It is not a textual
preprocessor include: each file has isolated preprocessor state, is included
exactly once, and declares `implementing nrc`. Only the primary `nrc.slang`
file owns the `#language slang 2026` directive and can be imported.

`nrc_renderer_adapter.h.slang` is a textual `#include`. The renderer's material
system is preprocessor-configured, whereas imported Slang modules have isolated
preprocessor state, so the adapter owns the set-3 resources and translates
renderer values into NRC semantic types. Bound resources and pass entry points
live in the root shaders rather than in the module.

## Typed records and encoders

`NrcRecord<TPayload>` captures the semantic vertex prefix shared by query and
training records. `QueryRecord` and `TrainingRecord` are aliases specialized
with their respective payloads. Under Vulkan scalar layout their buffer strides
are 80 and 140 bytes.

`IEncoder<TInput>` defines one compile-time encoder interface with an associated
output type. `FrequencyEncoder<N>` and `OneBlobEncoder<N>` are reusable scalar
implementations. `NrcFeatureEncoder` composes them and is the sole owner of the
private 64-lane network packing. Encoder selection resolves at compile time,
with no runtime interface dispatch.

## Generic RTXNS network

The RTXNS facade is a stateless generic value type:

```slang
Network<InputCount, HiddenCount, OutputCount, HiddenLayerCount>
```

Its `inference`, `forward`, and `backward` methods hide cooperative-vector types.
The backward method uses Slang autodiff with a custom linear-layer derivative
that accumulates matrix and bias gradients in FP32. Forward evaluation and
reverse matrix-vector products retain RTXNS's FP16 cooperative-vector path.
`NrcNetwork` aliases the
`Network<64, 64, 3, 5>` configuration and adds the semantic feature and RGB
overloads used by the Chapter 7 passes. RTXNS is the only backend, so the module
binds to it directly rather than through a backend interface.

## File map

| File | Contents |
|---|---|
| `nrc.slang` | Module root. Owns the `#language slang 2026` directive, imports the RTXNS backend, and assembles the implementation files below with `__include`. The only importable NRC file. |
| `nrc/records.slang` | Semantic and buffer-facing records: `NrcVertexFeatures`, the generic `NrcRecord<TPayload>`, its `QueryRecord` and `TrainingRecord` aliases, and the packed training-record flag constants. |
| `nrc/encoders.slang` | Reusable encoder vocabulary, with nothing NRC-specific: the `IEncoding` and `IEncoder` interfaces, `ScalarEncoding<N>`, `FrequencyEncoder<N>`, and `OneBlobEncoder<N>`. |
| `nrc/features.slang` | NRC's concrete feature layout. `NrcFeatureEncoder` composes the generic encoders and is the sole owner of the private 64-lane network packing (`kEncodedFeatureCount`). |
| `nrc/estimator.slang` | Estimator and training policy: path-spread heuristics, training-ray and unbiased-RR selection, and construction and completion of query and training records. |
| `nrc/network.slang` | The generic `Network<InputCount, HiddenCount, OutputCount, HiddenLayerCount>` facade over RTXNS, exposing `inference`, `forward`, and `backward`, plus the `NrcNetwork` alias for the default 64/64/3/5 configuration. |
| `nrc_renderer_adapter.h.slang` | The renderer adapter, and the only textual `#include`. Owns the descriptor set 3 bindings and translates renderer values into NRC semantic types. |
| `nrc_cache_query.comp.slang`, `nrc_record_complete.comp.slang`, `nrc_train.comp.slang` | Compute entry points that `import nrc`. Each is a thin `main` plus its explicit Vulkan bindings. |
| `nrc_optimizer.comp.slang`, `nrc_compose.comp.slang` | Compute entry points that work on raw weight and radiance buffers, so they do not import the module. |
| `shaders/tests/test_mlp_facade.comp.slang` | Compile test built by the `nrc_shader_compile_test` target. Instantiates the record, encoder, inference, forward, and backward paths with the production compiler. |
| `tests/test_nrc_transport.comp.slang`, `tests/test_nrc_transport.cpp` | Executable Vulkan regression fixtures that call the production estimator helpers and compare their results with analytic transport values. |
| `tests/test_nrc_precision.cpp` | Executes the production training and optimizer shaders to check full-batch FP32 accumulation and persistent Adam/EMA precision. |

## Transport contract

The cache predicts local scattered radiance, including the queried surface's
direct lighting and excluding that surface's own emission. The rendering
prefix includes the emission before choosing a cache query; it stops before
the queried surface's next-event estimation or BSDF sample. This boundary
prevents counting direct illumination twice when the cache result is composed.

Each pending training record starts with unit **local** throughput. Subsequent
BSDF, medium-transmittance, and roulette-survival factors update that record's
throughput, and local lighting contributions accumulate through it. Training
never divides camera-space radiance by camera throughput. Consequently, a
record created after a colored bounce can still learn channels that earlier
bounces removed from the camera contribution.

Rendering-prefix spread and training-suffix spread are separate. A training
path restarts the spread sum at its rendering query, and a later eligible
vertex becomes its bootstrap endpoint before that vertex's lighting or
roulette. One sixteenth of selected training paths ignore the suffix spread
termination policy and run to natural termination with roulette. Their targets
contain no cache tail. The rendering depth limit does not truncate these
selected training suffixes. A finite recording stack limits the number of
records, not the validity of an endpoint or the length of the traced suffix.

Cache positions use the scene's normalized bounds. A surface outside those
bounds continues through the path tracer; its position is never clamped into
an unrelated cache location. Delta BSDF samples contribute zero spread,
including the renderer's negative `DIRAC` PDF sentinel.

Bootstrap target completion always reads primary weights. EMA weights are
reserved for rendering. Targets are factored by the record's reflectance
without a fixed clamp in normalized space; the renderer's configurable firefly
clamp acts on the final physical pixel estimate. Normalization and
reconstruction share the same guarded reflectance factor, including for
channels below `1e-3`. These contracts follow the
prefix/suffix and training-feedback separation in
[the NRC paper, sections 3.2-3.4](https://d1qx31qr3h6wln.cloudfront.net/publications/mueller21realtime.pdf).

## Interpreting Cache Debug and EMA

Cache Debug queries the first eligible surface and displays its reconstructed
cache radiance. Normal NRC waits for the path-spread heuristic. The early
debug query exposes approximation errors that later rough bounces can average
out, including bands from the positional frequency encoding. The paper
discusses this limitation in section 6, figure 12, and section 7. A striped
debug image alone does not establish a broken encoder or optimizer; compare
the final NRC image with a matching path-traced reference as well.

EMA averages network parameters over training updates. It can reduce temporal
fluctuations, but it does not spatially filter an image or guarantee removal of
encoding bands. Both primary and EMA parameters are maintained while training,
regardless of which set is selected for rendering. To compare them, train once,
enable Lock, then capture each selection with fresh accumulation. Changing Lock,
Use EMA, or the view mode resets accumulation without reinitializing the network.
Re-Train reinitializes the network. Max Iterations also stops training when it
stops rendered frames.

For lighting comparisons, keep the camera and tonemapper fixed and use a
sufficient path depth for the reference. A per-sample firefly clamp can remove
more energy from a noisy path-traced estimate than a cache estimate, even at the
same threshold. Use a high threshold for this diagnostic; zero currently makes
positive radiance black rather than disabling the clamp.

## Training precision and GPU ordering

The network consumes FP16 parameters and activations. Parameter gradients,
Adam master parameters, both Adam moments, and EMA master parameters persist
in FP32. Gradient accumulation uses its own driver-queried FP32 matrix layout;
its byte offsets are independent of the FP16 inference layout. Batch averaging
occurs before converting the upstream gradient to FP16, and the optimizer
removes the loss scale. Matrix conversion publishes the FP32 master state to
the FP16 buffers without feeding quantization back into Adam or EMA.

Gradient clearing, backward accumulation, Adam updates, matrix conversions,
and inference require explicit Vulkan dependencies. Matrix conversion uses
`VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV`. Primary and EMA
query descriptor sets are allocated separately so switching the UI selection
does not rewrite an in-flight descriptor. Counter readbacks use slots gated by
completed frame timeline values.

Minibatches use one seeded permutation of the populated record pool per frame.
Every record is visited before repeating the pool to fill the fixed-size
batches. Cycle walking restricts a permutation to the actual count, so partially
filled pools retain complete coverage. The earlier wrapped multiply followed by
modulo could omit more than half a partial pool.

The GPU also tracks the number of nonempty training steps. Empty record frames
preserve Adam's step, both moments, primary weights, and EMA weights. Supplying
zero gradients alone is insufficient because existing momentum would continue
to move the network. This matters when both views use None or no eligible
surfaces emit records.

## Validation

From the Chapter 7 directory, with a configured build tree:

```powershell
cmake --build build --config Release --target nrc_shader_compile_test
cmake --build build --config Release --target nrc_transport_check
cmake --build build --config Release --target nrc_precision_check
cmake --build build --config Release --target nrc_sampler_check
```

`nrc_transport_check` compiles the real `nrc` module into a small compute shader
and executes it on a Vulkan 1.3 compute device. It requires no ray-tracing or
cooperative-vector feature. The fixtures check local multi-bounce RGB targets,
bootstrap throughput and endpoint data, zero/tiny-channel colored prefixes,
roulette compensation, natural termination, delta spread, suffix policy, and
out-of-domain rejection against hand-solved results. A dark-material fixture
checks that reflectance factorization preserves physical radiance. Inputs come from a
host buffer, so the shader calculations execute on the GPU.

`nrc_precision_check` requires a cooperative-vector training device. It runs a
16,384-sample batch through the production training shader and checks the
output bias and matrix gradients against an analytic result. It then executes
an independent nonuniform network fixture that checks 17,411 gradients across
all six layers, including active and inactive hidden ReLUs and transposed
backward matrix products. Its inference output is checked against a CPU forward
calculation through the production record-completion shader. The test executes
100 production optimizer steps to verify that FP32 master parameters and EMA
retain changes smaller than an FP16 rounding step, and checks that nonfinite
gradients preserve the existing optimizer state.
An empty-record regression checks that nonzero momentum, unequal EMA weights,
and the optimizer step remain unchanged across empty batches, then resume at
the next actual step.
The production record-completion shader is also exercised with dark and zero
reflectance channels: valid physical radiance yields finite normalized targets
above the old clamp of 20.

`nrc_sampler_check` checks complete coverage, balanced repetitions, index bounds,
and seed variation using the production GPU sampling helper. Its populations
include 49,152, 55,000, and 60,000 records, where the previous partial-pool
shuffle omitted records, as well as small and power-of-two populations.

The tests check the estimator helpers, not every branch of the full renderer.
Renderer validation additionally needs controlled diffuse, mirror,
transmission, and volume scenes, with Vulkan synchronization validation enabled.
Compare NRC with a converged path-traced reference at the same camera and
exposure, and record how many unlocked training frames preceded the capture.

The renderer exposes `--nrcLeftView` and `--nrcRightView` (`0` path tracer,
`1` NRC, `2` cache debug), `--nrcUseEma`, and `--ptUseSER` for repeatable
captures. For example, after building `Chapter7`:

```powershell
$env:VK_LAYER_VALIDATE_SYNC = '1'
& _bin/Release/Chapter7.exe --headless --frames 100000 --maxFrames 512 `
  --size 640 360 --scenefile Sponza/MyScene.gltf --ptTechnique 1 `
  --nrc 1 --nrcLeftView 2 --nrcRightView 1 --nrcUseEma 1 `
  --ptAccumulate 1 --vvl 1 --dlssEnable 0 --optixEnable 0 `
  --output "$env:TEMP/nrc-sponza.png"
```

Headless loop iterations include asynchronous pipeline creation. Confirm the
log reports nonzero NRC training/query counts; a cold pipeline can consume
the loop before rendering starts. `--maxFrames` controls rendered iterations.
Use `--ptAccumulate 0` to inspect the final 1-spp frame without averaging early
training estimates into the image.
