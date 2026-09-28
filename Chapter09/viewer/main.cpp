/*
 * Listing 9.2 -- A real-time Gaussian splat viewer in LightweightVK
 * =================================================================
 * Every frame: sort the splats back to front, draw one instanced quad each,
 * let fixed-function blending composite them. The vertex shader does the same
 * mathematics as `project_gaussians` in Listing 9.1 -- a 3D Gaussian projects
 * to an ellipse with covariance J Sigma J^T -- and the fragment shader
 * evaluates the Gaussian per pixel from its inverse.
 *
 * Window, context, camera, input, main loop and screenshot are `VulkanApp`,
 * the harness LightweightVK's samples run on; RPly reads the model,
 * nlohmann/json the camera metadata, glm the math, std::sort the sort. What
 * is left is the shaders and the splat-specific glue. Five things in that
 * glue are not obvious and each cost a debugging session: marked GOTCHA.
 */

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <execution>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <VulkanApp.h> // LightweightVK's own sample harness
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <rply.h>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kRingSize = 3; // order buffers in flight
constexpr lvk::Format kHdrFormat = lvk::Format_RGBA_F16;

/* One splat in the shader's std430 layout. 240 bytes each -- 100 MiB for a
 * 450,000-splat scene, which is why splatting is memory-bound first. */
struct alignas(16) GpuSplat {
  glm::vec4 posOpacity; // xyz = position, w = opacity, already through sigmoid
  glm::vec4 covA; // 3D covariance xx, xy, xz
  glm::vec4 covB; // 3D covariance yy, yz, zz
  float sh[48]; // 16 SH coefficients x 3 channels, coefficient-major
};
static_assert(sizeof(GpuSplat) == 240);

// ---------------------------------------------------------------------------
// Shaders. LightweightVK prepends `#version` and the buffer-reference
// extensions to any GLSL that does not declare its own version.
// ---------------------------------------------------------------------------

const char* kSplatVS = R"(
struct Splat {
  vec4 posOpacity;
  vec4 covA;
  vec4 covB;
  float sh[48];
};
layout(std430, buffer_reference) readonly buffer Splats { Splat items[]; };
layout(std430, buffer_reference) readonly buffer Order  { uint  items[]; };

layout(push_constant) uniform Constants {
  mat4 view;         // world -> camera: x right, y DOWN, z forward (OpenCV)
  Splats splats;
  Order  order;
  vec2 focal;        // pixels
  vec2 viewport;     // pixels
  vec4 cameraPos;
  int  shDegree;
} pc;

layout(location = 0) out vec2 vDelta;   // pixel offset from the splat center
layout(location = 1) out vec3 vConic;   // inverse 2D covariance: xx, xy, yy
layout(location = 2) out vec4 vColor;

const vec2 kCorner[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(-1, 1),
                                vec2(-1, 1), vec2(1, -1), vec2(1, 1));

const float kC0 = 0.28209479177387814;
const float kC1 = 0.4886025119029199;
const float kC2[5] = float[5](1.0925484305920792, -1.0925484305920792,
                              0.31539156525252005, -1.0925484305920792,
                              0.5462742152960396);
const float kC3[7] = float[7](-0.5900435899266435, 2.890611442640554,
                              -0.4570457994644658, 0.3731763325901154,
                              -0.4570457994644658, 1.445305721320277,
                              -0.5900435899266435);

vec3 sh(Splat s, int k) { return vec3(s.sh[3*k], s.sh[3*k+1], s.sh[3*k+2]); }

// Identical to eval_sh() in Listing 9.1 -- same basis, same signs. A sign out
// of place here is a scene that is subtly the wrong color from every angle
// except the ones it was trained on.
vec3 evalSH(Splat s, vec3 d, int degree) {
  vec3 c = kC0 * sh(s, 0);
  if (degree >= 1) {
    c += -kC1*d.y*sh(s,1) + kC1*d.z*sh(s,2) - kC1*d.x*sh(s,3);
    if (degree >= 2) {
      float xx=d.x*d.x, yy=d.y*d.y, zz=d.z*d.z, xy=d.x*d.y, yz=d.y*d.z, xz=d.x*d.z;
      c += kC2[0]*xy*sh(s,4) + kC2[1]*yz*sh(s,5)
         + kC2[2]*(2.0*zz-xx-yy)*sh(s,6) + kC2[3]*xz*sh(s,7)
         + kC2[4]*(xx-yy)*sh(s,8);
      if (degree >= 3) {
        c += kC3[0]*d.y*(3.0*xx-yy)*sh(s,9) + kC3[1]*xy*d.z*sh(s,10)
           + kC3[2]*d.y*(4.0*zz-xx-yy)*sh(s,11)
           + kC3[3]*d.z*(2.0*zz-3.0*xx-3.0*yy)*sh(s,12)
           + kC3[4]*d.x*(4.0*zz-xx-yy)*sh(s,13)
           + kC3[5]*d.z*(xx-yy)*sh(s,14) + kC3[6]*d.x*(xx-3.0*yy)*sh(s,15);
      }
    }
  }
  return max(c + 0.5, vec3(0.0));
}

void main() {
  Splat s = pc.splats.items[pc.order.items[gl_InstanceIndex]];
  const vec3 cam = (pc.view * vec4(s.posOpacity.xyz, 1.0)).xyz;
  if (cam.z < 0.01) { gl_Position = vec4(0, 0, 2, 1); return; }

  // GOTCHA 1: clamp the Jacobian. The projection is linearized at the splat's
  // center; far off to the side that diverges, the covariance explodes and the
  // splat smears over the frame as haze. An object capture may contain no such
  // splat, a forward-facing outdoor scene thousands. Evaluate at the frustum
  // edge instead (1.3, as in the reference implementation).
  const vec2 lim = 1.3 * 0.5 * pc.viewport / pc.focal;
  const vec2 tc = clamp(cam.xy / cam.z, -lim, lim) * cam.z;
  const mat3 J = mat3(pc.focal.x / cam.z, 0.0, 0.0,
                      0.0, pc.focal.y / cam.z, 0.0,
                      -pc.focal.x * tc.x / (cam.z * cam.z),
                      -pc.focal.y * tc.y / (cam.z * cam.z), 0.0);
  const mat3 T = J * mat3(pc.view);
  const mat3 sigma = mat3(s.covA.x, s.covA.y, s.covA.z,
                          s.covA.y, s.covB.x, s.covB.y,
                          s.covA.z, s.covB.y, s.covB.z);
  const mat3 cov = T * sigma * transpose(T);

  // The low-pass dilation the trainer also applies: a splat narrower than a
  // pixel must not fall between the sample points.
  const float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
  const float det = a * c - b * b;
  if (det <= 0.0) { gl_Position = vec4(0, 0, 2, 1); return; }
  vConic = vec3(c, -b, a) / det;

  // Fit the quad to the ellipse's 3-sigma axes rather than to a bounding
  // square: closed-form eigen-decomposition of a symmetric 2x2, and roughly
  // half the fragments to shade and reject.
  const float mid = 0.5 * (a + c);
  const float disc = sqrt(max(mid * mid - det, 0.0));
  const vec2 axis = abs(b) > 1e-6 ? normalize(vec2(b, mid + disc - a))
                                  : (a >= c ? vec2(1, 0) : vec2(0, 1));
  vDelta = kCorner[gl_VertexIndex].x * 3.0 * sqrt(mid + disc) * axis
         + kCorner[gl_VertexIndex].y * 3.0 * sqrt(max(mid - disc, 0.0))
           * vec2(-axis.y, axis.x);

  // GOTCHA 2: flip Y. LightweightVK binds a NEGATIVE-height viewport, so clip
  // space is +Y up, while this camera -- like OpenCV and the trainer -- is +Y
  // down. Miss it and the scene renders perfectly, upside down. Only the
  // position flips; vDelta stays in y-down space so the conic stays valid.
  const vec2 screen = vec2(pc.focal.x * cam.x, pc.focal.y * cam.y) / cam.z + vDelta;
  gl_Position = vec4(vec2(screen.x, -screen.y) * 2.0 / pc.viewport, 0.5, 1.0);

  vColor = vec4(evalSH(s, normalize(s.posOpacity.xyz - pc.cameraPos.xyz), pc.shDegree),
                s.posOpacity.w);
}
)";

const char* kSplatFS = R"(
layout(location = 0) in vec2 vDelta;
layout(location = 1) in vec3 vConic;
layout(location = 2) in vec4 vColor;
layout(location = 0) out vec4 out_FragColor;

void main() {
  const float power = -0.5 * (vConic.x * vDelta.x * vDelta.x +
                              2.0 * vConic.y * vDelta.x * vDelta.y +
                              vConic.z * vDelta.y * vDelta.y);
  const float alpha = min(0.99, vColor.a * exp(power));
  // Far below 1/255: hundreds of faint tails sum to a haze worth several
  // levels, and the float target can hold them.
  if (alpha < 1.0 / 65536.0) discard;
  out_FragColor = vec4(vColor.rgb * alpha, alpha);   // premultiplied
}
)";

// Full-screen triangle copying the HDR buffer to the swapchain. The UV comes
// from gl_FragCoord, sidestepping GOTCHA 2 entirely.
const char* kResolveVS = R"(
void main() {
  const vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* kResolveFS = R"(
layout(push_constant) uniform Constants { uint tex; uint smp; vec2 viewport; } pc;
layout(location = 0) out vec4 out_FragColor;
void main() {
  out_FragColor = vec4(textureBindless2D(pc.tex, pc.smp,
                                         gl_FragCoord.xy / pc.viewport).rgb, 1.0);
}
)";

struct alignas(16) PushConstants {
  glm::mat4 view;
  uint64_t splats = 0;
  uint64_t order = 0;
  glm::vec2 focal{0.0f};
  glm::vec2 viewport{0.0f};
  glm::vec4 cameraPos{0.0f};
  int32_t shDegree = 3;
  int32_t pad[3] = {};
};
static_assert(sizeof(PushConstants) == 128);

struct ResolveConstants {
  uint32_t tex = 0, smp = 0;
  glm::vec2 viewport{0.0f};
};

// ---------------------------------------------------------------------------
// Loading the model, with RPly
// ---------------------------------------------------------------------------

/*
 * A splat PLY is a vertex element with ~62 float properties each: position, an
 * unused normal, 48 SH coefficients, opacity, three scales, a quaternion.
 * RPly delivers them one at a time through a callback -- more code than a
 * memcpy of the trainer's own layout, and it buys the ASCII, big-endian and
 * half-float variants other tools write.
 */
struct PlyTable {
  std::vector<float> values; // instance-major: values[i * stride + slot]
  size_t stride = 0;
};

int plyValueCallback(p_ply_argument argument) {
  long instance = 0, slot = 0;
  void* table = nullptr;
  ply_get_argument_element(argument, nullptr, &instance);
  ply_get_argument_user_data(argument, &table, &slot);
  PlyTable* t = static_cast<PlyTable*>(table);
  t->values[size_t(instance) * t->stride + size_t(slot)] = float(ply_get_argument_value(argument));
  return 1;
}

std::string withThousands(long n) {
  std::string text = std::to_string(n);
  for (int i = int(text.size()) - 3; i > 0; i -= 3)
    text.insert(size_t(i), ",");
  return text;
}

/* The `vertex` element's properties, in file order. */
struct PlyHeader {
  std::unordered_map<std::string, long> slotOf;
  std::vector<std::string> names;
  long count = 0;
};

/* Read the header of an open PLY and check that it describes splats.
 * `path` is needed only to sanity-check the declared vertex count. */
bool readPlyHeader(p_ply ply, const fs::path& path, PlyHeader& out, std::string& error) {
  for (p_ply_element el = nullptr; (el = ply_get_next_element(ply, el));) {
    const char* elementName = nullptr;
    long instances = 0;
    ply_get_element_info(el, &elementName, &instances);
    if (std::strcmp(elementName, "vertex") != 0)
      continue;
    out.count = instances;
    for (p_ply_property pr = nullptr; (pr = ply_get_next_property(el, pr));) {
      const char* propertyName = nullptr;
      ply_get_property_info(pr, &propertyName, nullptr, nullptr, nullptr);
      out.slotOf[propertyName] = long(out.names.size());
      out.names.emplace_back(propertyName);
    }
  }
  // Every property read below must be present, not just the first of each
  // group: `slotOf[missing]` would insert a zero and silently read column 0.
  for (const char* need : {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                           "f_dc_0", "f_dc_1", "f_dc_2"}) {
    if (!out.slotOf.count(need)) {
      error = std::string("PLY has no '") + need + "' property";
      return false;
    }
  }
  // The declared count drives the allocation below, and the smallest property
  // is one byte: a file shorter than count * properties cannot hold what it
  // claims. Unchecked, a corrupt header asks for hundreds of gigabytes and the
  // bad_alloc escapes to std::terminate.
  std::error_code ec;
  const uintmax_t bytes = fs::file_size(path, ec);
  if (out.count <= 0 || (!ec && uintmax_t(out.count) * out.names.size() > bytes)) {
    error = "PLY header claims " + withThousands(out.count) + " vertices, which the file cannot hold";
    return false;
  }
  return true;
}

/* Splat count of `path`, or 0 if it is not a splat PLY. Header only, so the
 * picker can probe a directory in the time one model takes to load. */
long probePly(const fs::path& path) {
  p_ply ply = ply_open(path.string().c_str(), nullptr, 0, nullptr);
  if (!ply)
    return 0;
  PlyHeader header;
  std::string ignored;
  const bool ok = ply_read_header(ply) && readPlyHeader(ply, path, header, ignored);
  ply_close(ply);
  return ok ? header.count : 0;
}

std::vector<GpuSplat> loadPly(const fs::path& path, std::string& error) {
  p_ply ply = ply_open(path.string().c_str(), nullptr, 0, nullptr);
  if (!ply || !ply_read_header(ply)) {
    error = "cannot read " + path.string() + " as a PLY file";
    if (ply)
      ply_close(ply);
    return {};
  }

  PlyHeader header;
  if (!readPlyHeader(ply, path, header, error)) {
    ply_close(ply);
    return {};
  }
  std::unordered_map<std::string, long>& slotOf = header.slotOf; // `[]` is non-const
  const std::vector<std::string>& names = header.names;
  const long count = header.count;

  PlyTable table{std::vector<float>(size_t(count) * names.size(), 0.0f), names.size()};
  for (const std::string& name : names)
    ply_set_read_cb(ply, "vertex", name.c_str(), plyValueCallback, &table, slotOf[name]);
  const int ok = ply_read(ply);
  ply_close(ply);
  if (!ok) {
    error = "PLY body is truncated or malformed";
    return {};
  }

  size_t rest = 0; // 3 * (SH bands above 0), so 0, 9, 24, 45, 72 ...
  while (slotOf.count("f_rest_" + std::to_string(rest)))
    rest++;
  const size_t inFile = rest / 3; // higher-band coefficients per channel
  // `sh` holds degree 3, which is 15. More bands is not an error -- they
  // cannot be shown -- but uncapped the loop below corrupts the heap.
  const size_t perChannel = std::min<size_t>(inFile, 15);

  // Look every property up BY NAME: `f_rest_i` sits `i` slots after
  // `f_rest_0` for the reference writer, but not for a tool that emits
  // properties in name order, where f_rest_10 follows f_rest_1 -- and such a
  // file then renders with quietly wrong view-dependent color.
  const long cx = slotOf["x"], cy = slotOf["y"], cz = slotOf["z"], co = slotOf["opacity"];
  long cs[3], cr[4], cd[3];
  for (int i = 0; i < 3; i++)
    cs[i] = slotOf["scale_" + std::to_string(i)];
  for (int i = 0; i < 4; i++)
    cr[i] = slotOf["rot_" + std::to_string(i)];
  for (int i = 0; i < 3; i++)
    cd[i] = slotOf["f_dc_" + std::to_string(i)];
  std::vector<long> cf(rest);
  for (size_t i = 0; i < rest; i++)
    cf[i] = slotOf["f_rest_" + std::to_string(i)];

  std::vector<GpuSplat> splats(static_cast<size_t>(count));
  for (size_t i = 0; i < splats.size(); i++) {
    const float* r = table.values.data() + i * table.stride;
    GpuSplat& g = splats[i];
    g.posOpacity = glm::vec4(r[cx], r[cy], r[cz], 1.0f / (1.0f + std::exp(-r[co])));

    const glm::mat3 M = glm::mat3_cast(glm::normalize(glm::quat(r[cr[0]], r[cr[1]], r[cr[2]], r[cr[3]]))) *
                        glm::mat3(std::exp(r[cs[0]]), 0, 0, 0, std::exp(r[cs[1]]), 0, 0, 0, std::exp(r[cs[2]]));
    const glm::mat3 sigma = M * glm::transpose(M);
    g.covA = glm::vec4(sigma[0][0], sigma[1][0], sigma[2][0], 0.0f);
    g.covB = glm::vec4(sigma[1][1], sigma[2][1], sigma[2][2], 0.0f);

    // The file stores the higher bands channel-major (all reds, then greens,
    // then blues); the shader wants them coefficient-major.
    for (size_t c = 0; c < 3; c++) {
      g.sh[c] = r[cd[c]];
      for (size_t k = 0; k < perChannel; k++)
        g.sh[3 * (k + 1) + c] = r[cf[c * inFile + k]];
    }
  }
  return splats;
}

// ---------------------------------------------------------------------------
// Where to point the camera
// ---------------------------------------------------------------------------

/*
 * A PLY carries no camera information, so we read the `cameras.json` beside
 * it. Guessing instead looks like a rendering bug: for a forward-facing
 * capture the guess lands outside the volume the cameras occupied, where the
 * Gaussians -- needles fitted to be seen end-on -- are seen edge-on and the
 * scene looks like shattered glass.
 */
struct SceneInfo {
  glm::vec3 eye{0, 0, -3}, target{0, 0, 0}, up{0, -1, 0};
  // Where gravity points, which is not the same as the opening camera's own
  // up axis: see `gravityFrom` below. It steers navigation only, so that the
  // opening view stays exactly the camera `cameras.json` describes.
  glm::vec3 gravity{0, -1, 0};
  float fovYDegrees = 45.0f;

  /*
   * Take the median camera: captures start and end at the edges of the
   * trajectory, and the middle usually points at whatever mattered.
   */
  bool loadReferenceCameras(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
      return false;
    const nlohmann::json cameras = nlohmann::json::parse(file, nullptr, /*allow_exceptions=*/false);
    if (!cameras.is_array() || cameras.empty())
      return false;

    const nlohmann::json& c = cameras[cameras.size() / 2];
    std::vector<float> pos;
    std::vector<std::vector<float>> rot;
    float fy = 0.0f, height = 0.0f;
    try {
      pos = c.at("position").get<std::vector<float>>();
      rot = c.at("rotation").get<std::vector<std::vector<float>>>();
      fy = c.at("fy").get<float>();
      height = c.at("height").get<float>();
    } catch (const nlohmann::json::exception&) {
      return false;
    }
    // Every row, not just the first: `rot[1][1]` below reads out of bounds on
    // a file whose second row is short, and the garbage becomes the camera.
    if (pos.size() != 3 || rot.size() != 3 || rot[0].size() != 3 || rot[1].size() != 3 || rot[2].size() != 3 ||
        fy <= 0.0f || height <= 0.0f)
      return false;

    // `rotation` is camera-to-world, row-major, so its COLUMNS are the
    // camera's axes in world space -- in the same OpenCV convention we use,
    // with +Y down the image.
    eye = glm::vec3(pos[0], pos[1], pos[2]);
    up = -glm::vec3(rot[0][1], rot[1][1], rot[2][1]);
    target = eye + glm::vec3(rot[0][2], rot[1][2], rot[2][2]) * 3.0f;
    fovYDegrees = glm::degrees(2.0f * std::atan(height / (2.0f * fy)));
    gravity = gravityFrom(cameras, up);
    return true;
  }

  /*
   * Up, estimated from every camera instead of from one of them.
   *
   * `up_` is the axis the first-person camera re-levels against on every mouse move, so getting it wrong tilts the whole
   * scene as soon as you look around. One camera's own up axis is a poor source: a camera aimed at the floor has a
   * horizontal up axis, and the still-photo capture in this chapter pitches down 38 degrees on average.
   *
   * This is the re-levelling axis only. The opening orientation still comes from the reference camera, so the image
   * `--screenshot` writes is the one `cameras.json` asks for, and Listing 9.4 keeps measuring the rasterizer rather than
   * our taste in cameras.
   *
   * A phone held in landscape keeps its RIGHT axis horizontal whatever the pitch, so up is the direction most nearly
   * orthogonal to every right axis: the eigenvector of sum(r r^T) with the smallest eigenvalue. Power iteration on
   * trace(M) I - M finds it, starting from `fallback` so the sign is settled too. If the photographer rolled the camera the
   * assumption fails, and the measured roll says so.
   */
  static glm::vec3 gravityFrom(const nlohmann::json& cameras, const glm::vec3& fallback) {
    std::vector<glm::vec3> rights;
    for (const nlohmann::json& cam : cameras) {
      std::vector<std::vector<float>> r;
      try {
        r = cam.at("rotation").get<std::vector<std::vector<float>>>();
      } catch (const nlohmann::json::exception&) {
        continue;
      }
      if (r.size() != 3 || r[0].size() != 3 || r[1].size() != 3 || r[2].size() != 3)
        continue;
      rights.emplace_back(r[0][0], r[1][0], r[2][0]);
    }
    if (rights.size() < 8)
      return fallback;

    glm::mat3 m(0.0f);
    for (const glm::vec3& r : rights)
      m += glm::outerProduct(r, r);
    const glm::mat3 b = glm::mat3(m[0][0] + m[1][1] + m[2][2]) - m;

    glm::vec3 v = fallback;
    for (int i = 0; i < 64; i++) {
      const glm::vec3 next = b * v;
      const float length = glm::length(next);
      if (length < 1e-12f)
        return fallback;
      v = next / length;
    }
    if (glm::dot(v, fallback) < 0.0f)
      v = -v;

    float roll = 0.0f;
    for (const glm::vec3& r : rights)
      roll += glm::degrees(std::asin(std::min(1.0f, std::abs(glm::dot(r, v)))));
    roll /= float(rights.size());
    if (roll > 15.0f) {
      printf("up: camera roll averages %.0f deg, so gravity cannot be estimated; using the opening camera's up\n", roll);
      return fallback;
    }
    printf("up: %.1f deg from the opening camera's own up axis (mean camera roll %.1f deg)\n",
           glm::degrees(std::acos(std::min(1.0f, std::abs(glm::dot(v, fallback))))),
           roll);
    return v;
  }
};

/*
 * The `cameras.json` that belongs to `ply`, or an empty path. Only the two
 * layouts this chapter supports: beside the model, as Part 1 writes it, and
 * two levels up, where the reference trainer puts it relative to
 * `<model>/point_cloud/iteration_N/`. Searching further up finds unrelated
 * files -- a model in `work/probe/` would open with `work/cameras.json`, the
 * poses of a different capture, and that looks like a rendering bug.
 */
fs::path camerasFor(const fs::path& ply) {
  std::error_code ec;
  const fs::path dir = ply.parent_path();
  for (const fs::path& candidate : {dir / "cameras.json", dir.parent_path().parent_path() / "cameras.json"})
    if (fs::exists(candidate, ec))
      return candidate;
  return {};
}

/* First existing path among `names`, searching `dir` and up to four parents. */
fs::path findNearby(fs::path dir, std::initializer_list<const char*> names) {
  std::error_code ec;
  for (int level = 0; level < 5 && !dir.empty(); level++) {
    for (const char* name : names)
      if (fs::exists(dir / name, ec))
        return dir / name;
    const fs::path parent = dir.parent_path();
    if (parent == dir)
      break;
    dir = parent;
  }
  return {};
}

// ---------------------------------------------------------------------------
// Finding the models to offer
// ---------------------------------------------------------------------------

/* One row of the startup dialog. */
struct Candidate {
  fs::path path;          // absolute, canonical -- what actually gets loaded
  std::string display;    // relative to the directory it was found under
  std::string splatsText; // "458,341"
  bool hasCameras = false;
};

/*
 * Splat models directly inside `root`, plus the reference trainer's
 * <model>/point_cloud/iteration_N/point_cloud.ply layout one level down;
 * `base` is only what the displayed path is relative to. Every hit is
 * validated by reading its header, so a PLY that is not a splat PLY is
 * skipped rather than offered and then rejected.
 */
void scanForModels(const fs::path& root, const fs::path& base, std::vector<Candidate>& out) {
  std::error_code ec;
  auto consider = [&](const fs::path& file) {
    const long count = probePly(file);
    if (!count)
      return;
    const fs::path canonical = fs::weakly_canonical(file, ec);
    for (const Candidate& existing : out)
      if (existing.path == canonical)
        return; // the walk below reaches the same directory more than once
    Candidate candidate;
    candidate.path = canonical;
    candidate.display = fs::relative(file, base, ec).generic_string();
    if (candidate.display.empty())
      candidate.display = file.filename().string();
    candidate.splatsText = withThousands(count);
    candidate.hasCameras = !camerasFor(file).empty();
    out.push_back(std::move(candidate));
  };

  for (const fs::directory_entry& entry : fs::directory_iterator(root, ec))
    if (entry.is_regular_file(ec) && entry.path().extension() == ".ply")
      consider(entry.path());

  for (const fs::directory_entry& model : fs::directory_iterator(root, ec)) {
    if (!model.is_directory(ec))
      continue;
    for (const fs::directory_entry& iteration : fs::directory_iterator(model.path() / "point_cloud", ec)) {
      if (!iteration.is_directory(ec))
        continue;
      for (const fs::directory_entry& entry : fs::directory_iterator(iteration.path(), ec))
        if (entry.is_regular_file(ec) && entry.path().extension() == ".ply")
          consider(entry.path());
    }
  }
}

/* Everything loadable in the directories `findNearby` would have searched. */
std::vector<Candidate> collectCandidates(std::initializer_list<fs::path> starts) {
  std::vector<Candidate> out;
  for (fs::path dir : starts) {
    for (int level = 0; level < 5 && !dir.empty(); level++) {
      // `work` is where the driver writes, from either the repository root or
      // `Chapter09` itself. The last two are where `deploy_deps.py` unpacks
      // the reference scenes from 3D Gaussian Splatting, so those show up in
      // the dialog too.
      for (const char* sub : {"", "work", "Chapter09/work", "deps/src/3dgs-pretrained", "Chapter09/deps/src/3dgs-pretrained"})
        scanForModels(*sub ? dir / sub : dir, dir, out);
      const fs::path parent = dir.parent_path();
      if (parent == dir)
        break;
      dir = parent;
    }
  }
  // This chapter's own output first, so the default offer -- and the model a
  // headless run picks -- is the one the old auto-detection would have found.
  std::stable_partition(out.begin(), out.end(), [](const Candidate& c) { return c.path.filename() == "splats.ply"; });
  return out;
}

// ---------------------------------------------------------------------------
// The depth sort, off the critical path
// ---------------------------------------------------------------------------

/*
 * GOTCHA 5: sort exactly. Quantizing depth to 16 bits sounds finer than
 * blending can resolve, but with 458,000 splats the buckets collide and
 * collisions blend in array order: 12.6 of 255 against the reference
 * renderer, versus 0.45 for an exact sort.
 *
 * But not every frame, and not on this thread. Since
 * dot(p - eye, forward) = dot(p, forward) - dot(eye, forward) and the second
 * term is common to every splat, the ORDER depends only on the view
 * direction -- flying never needs a re-sort, only looking around does. Then a
 * worker sorts while the frame that asked is already on screen and the result
 * lands a frame or two later, which is fine: a stale order is a slightly
 * wrong blend between splats nearly coincident in depth.
 */
class DepthSorter {
 public:
  // All four buffers are sized here and keep that size, so the swaps below
  // never allocate. `published_` must not start empty: the first swap would
  // hand the worker a zero-length buffer to write into.
  DepthSorter(const std::vector<GpuSplat>& splats, const glm::vec3& forward)
  : splats_(splats)
  , order_(splats.size())
  , published_(splats.size())
  , depth_(splats.size())
  , result_(splats.size()) {
    sort(forward); // frame one draws a correct order
    result_ = order_;
    assert(published_.size() == order_.size() && result_.size() == order_.size());
    haveResult_ = true;
    requested_ = forward;
    thread_ = std::thread([this] { run(); });
  }
  ~DepthSorter() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    wake_.notify_one();
    thread_.join();
  }

  /* Ask for a re-sort if the view has turned since the last request. */
  void request(const glm::vec3& forward) {
    if (glm::dot(forward, requested_) > 0.99999f)
      return; // under ~0.25 degrees: the order cannot have changed
    requested_ = forward;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_ = forward;
      havePending_ = true;
    }
    wake_.notify_one();
  }

  /* The newest completed order, or nullptr if nothing has finished since the
   * last call. The returned buffer belongs to the caller's thread. */
  const std::vector<uint32_t>* take() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!haveResult_)
      return nullptr;
    haveResult_ = false;
    published_.swap(result_);
    return &published_;
  }

 private:
  void sort(const glm::vec3& forward) {
    for (size_t i = 0; i < splats_.size(); i++) {
      const float z = glm::dot(glm::vec3(splats_[i].posOpacity), forward);
      // A NaN position -- a diverged training run produces them -- compares
      // false against everything, so `>` would stop being a strict weak
      // ordering and std::sort would be undefined. Send them to the back.
      depth_[i] = std::isnan(z) ? -std::numeric_limits<float>::infinity() : z;
      order_[i] = uint32_t(i);
    }
    std::sort(std::execution::par_unseq, order_.begin(), order_.end(), [this](uint32_t a, uint32_t b) { return depth_[a] > depth_[b]; });
  }

  void run() {
    for (;;) {
      glm::vec3 forward;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return havePending_ || quit_; });
        if (quit_)
          return;
        forward = pending_; // always the most recent request
        havePending_ = false;
      }
      sort(forward);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        result_.swap(order_); // no copy; order_ is overwritten next
        haveResult_ = true;
      }
    }
  }

  const std::vector<GpuSplat>& splats_;
  std::vector<uint32_t> order_; // worker
  std::vector<uint32_t> published_; // main thread
  std::vector<float> depth_; // worker
  std::vector<uint32_t> result_; // guarded: the handoff
  glm::vec3 requested_{0.0f}; // main thread only

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable wake_;
  glm::vec3 pending_{0.0f}; // guarded
  bool havePending_ = false, haveResult_ = false, quit_ = false;
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

SceneInfo g_home;
float g_sceneScale = 1.0f;
int g_shDegree = 3;

/* OpenGL view matrix (down -Z, +Y up) -> the OpenCV convention everything
 * else here uses (+Z forward, +Y down). Its own inverse. */
glm::mat4 toOpenCvView(const glm::mat4& gl) {
  glm::mat4 m = gl;
  for (int c = 0; c < 4; c++) {
    m[c][1] = -m[c][1];
    m[c][2] = -m[c][2];
  }
  return m;
}

} // namespace

// ---------------------------------------------------------------------------

VULKAN_APP_MAIN {
  // Positionals are the model and its cameras.json; every `--flag` belongs to
  // VulkanApp, which parses argv itself. Those taking a value must be named,
  // or `--width 800` would leave `800` looking like a path.
  std::vector<std::string> positional;
  std::string screenshot;
  bool headless = false;
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      positional.push_back(arg);
      continue;
    }
    if (arg == "--screenshot" || arg == "--screenshot-file" || arg == "--screenshot-frame")
      headless = true;
    for (const char* valued : {"--screenshot", "--screenshot-file", "--log-file", "--screenshot-frame", "--width", "--height"}) {
      if (arg == valued && i + 1 < argc) {
        if (arg == "--screenshot")
          screenshot = argv[i + 1];
        ++i;
        break;
      }
    }
  }

  // With no model on the command line the viewer asks instead of guessing, so
  // loading has to wait until the window exists: that is what `loadScene` is
  // for, and `drawPicker` below offers what `collectCandidates` found.
  fs::path ply = positional.empty() ? fs::path() : fs::path(positional[0]);
  const fs::path camerasArg = positional.size() > 1 ? fs::path(positional[1]) : fs::path();
  std::vector<Candidate> candidates;
  if (ply.empty()) {
    std::error_code ec;
    candidates = collectCandidates({fs::current_path(ec), fs::weakly_canonical(argv[0], ec).parent_path()});
    // Nobody is there to click during a screenshot run, so take the first
    // offer -- which is the model the old auto-detection would have loaded.
    if (headless && !candidates.empty())
      ply = candidates.front().path;
  }
  if (ply.empty() && !candidates.empty()) {
    printf("found %zu models; pick one in the window:\n", candidates.size());
    for (const Candidate& c : candidates)
      printf("  %-52s %12s splats%s\n", c.display.c_str(), c.splatsText.c_str(), c.hasCameras ? "" : "  (no cameras.json)");
  }
  if (ply.empty() && headless) {
    fprintf(stderr,
            "error: no splats.ply given and none found nearby\nusage: %s [splats.ply] [cameras.json] [--screenshot out.png]\n",
            argv[0]);
    return 1;
  }

  /*
   * The subdir fields point the harness at LightweightVK's data inside this
   * book's tree. VulkanApp searches for them only ABOVE THE CURRENT
   * DIRECTORY, so a viewer started elsewhere -- or double-clicked -- finds no
   * font and the dialog has nothing to draw with. Resolve them here, from the
   * executable as well, and pass absolute paths: `dir / subdir` leaves an
   * absolute subdir alone. These strings must outlive `app`.
   */
  std::string contentDir = "Chapter09/deps/src/lightweightvk/third-party/content/";
  std::string thirdPartyDir = "Chapter09/deps/src/lightweightvk/third-party/deps/src/";
  {
    std::error_code ec;
    for (const fs::path& from : {fs::current_path(ec), fs::weakly_canonical(argv[0], ec).parent_path()}) {
      const fs::path content = findNearby(from, {"deps/src/lightweightvk/third-party/content",
                                                 "Chapter09/deps/src/lightweightvk/third-party/content"});
      if (content.empty())
        continue;
      contentDir = content.string() + "/";
      thirdPartyDir = (content.parent_path() / "deps/src").string() + "/";
      break;
    }
  }

  /*
   * The camera is deliberately NOT configured here. It belongs to whichever
   * model is loaded, and that may not be chosen for several hundred frames.
   */
  VulkanAppConfig cfg{
      .width = -90, // 90% of the monitor work area
      .height = -90,
      .resizable = true,
      .screenshotFrameNumber = screenshot.empty() ? 0u : 1u,
      .screenshotFileName = screenshot.empty() ? "screenshot.png" : screenshot.c_str(),
      .contentSubdir = contentDir.c_str(),
      .thirdPartySubdir = thirdPartyDir.c_str(),
  };
#if defined(NDEBUG)
  // Validation is invaluable while writing a renderer and ruinous while
  // measuring one; keep it in debug builds only.
  cfg.contextConfig.enableValidation = false;
  cfg.contextConfig.enableValidationGpuAV = false;
#endif
  VULKAN_APP_DECLARE(app, cfg);
  lvk::IContext* ctx = app.ctx_.get();

  // VulkanApp binds WSAD, 1/2, Shift and Space already. Two additions: the
  // wheel dollies, and the brackets step the spherical-harmonic degree --
  // watch the highlights on metal flatten as bands are switched off.
  app.addScrollCallback([](GLFWwindow* w, double, double dy) {
    // A free-fly camera has no orbit distance to shrink, so the wheel dollies
    // along the view axis instead.
    CameraPositioner_FirstPerson& p = ((VulkanApp*)glfwGetWindowUserPointer(w))->positioner_;
    const glm::mat4 v = p.getViewMatrix();
    p.setPosition(p.getPosition() - glm::vec3(v[0][2], v[1][2], v[2][2]) * float(dy) * 0.1f * g_sceneScale);
  });
  app.addKeyCallback([](GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET) {
      g_shDegree = std::clamp(g_shDegree + (key == GLFW_KEY_RIGHT_BRACKET ? 1 : -1), 0, 3);
    }
  });

  // ---- resources that outlive the model ------------------------------------
  /*
   * GOTCHA 4: accumulate into a float target, not the swapchain. Blending
   * happens in the framebuffer, so 8 bits re-quantize the running color after
   * every splat -- and with a median opacity of 0.087, a pixel integrates
   * hundreds of them. The rounding compounds into visible streaks.
   */
  lvk::Holder<lvk::TextureHandle> hdr;
  auto ensureHdr = [&](uint32_t w, uint32_t h) {
    const lvk::Dimensions have = hdr.valid() ? ctx->getDimensions(hdr) : lvk::Dimensions{};
    if (have.width == w && have.height == h)
      return;
    ctx->wait({});
    hdr = ctx->createTexture({.type = lvk::TextureType_2D,
                              .format = kHdrFormat,
                              .dimensions = {w, h},
                              .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
                              .debugName = "Texture: HDR accumulation"});
  };

  lvk::Holder<lvk::SamplerHandle> sampler = ctx->createSampler({.wrapU = lvk::SamplerWrap_Clamp, .wrapV = lvk::SamplerWrap_Clamp});
  lvk::Holder<lvk::ShaderModuleHandle> vert = ctx->createShaderModule({kSplatVS, lvk::Stage_Vert, "Shader: splat (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag = ctx->createShaderModule({kSplatFS, lvk::Stage_Frag, "Shader: splat (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> rvert = ctx->createShaderModule({kResolveVS, lvk::Stage_Vert, "Shader: resolve (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> rfrag = ctx->createShaderModule({kResolveFS, lvk::Stage_Frag, "Shader: resolve (frag)"});

  lvk::Holder<lvk::RenderPipelineHandle> pipeline =
      ctx->createRenderPipeline({.smVert = vert,
                                 .smFrag = frag,
                                 // Back-to-front "over", with the source already premultiplied.
                                 .color = {{.format = kHdrFormat,
                                            .blendEnabled = true,
                                            .srcRGBBlendFactor = lvk::BlendFactor_One,
                                            .srcAlphaBlendFactor = lvk::BlendFactor_One,
                                            .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
                                            .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha}},
                                 .cullMode = lvk::CullMode_None,
                                 .debugName = "Pipeline: splats"});
  lvk::Holder<lvk::RenderPipelineHandle> resolve = ctx->createRenderPipeline({.smVert = rvert,
                                                                              .smFrag = rfrag,
                                                                              .color = {{.format = ctx->getSwapchainFormat()}},
                                                                              .cullMode = lvk::CullMode_None,
                                                                              .debugName = "Pipeline: resolve"});

  // ---- resources that belong to one model ----------------------------------
  // `drawSlot` holds the newest order the GPU can read; `uploadSlot` is where
  // the next one goes. They are never the same buffer, and a slot is only
  // rewritten once the submission that read it has retired.
  std::vector<GpuSplat> splats; // DepthSorter keeps a reference to this
  lvk::Holder<lvk::BufferHandle> splatBuffer;
  lvk::Holder<lvk::BufferHandle> orderBuffer[kRingSize];
  lvk::SubmitHandle inFlight[kRingSize];
  std::optional<DepthSorter> sorter; // empty until a model is chosen
  uint32_t drawSlot = 0, uploadSlot = 0;

  auto loadScene = [&](const fs::path& path, std::string& error) -> bool {
    std::vector<GpuSplat> loaded = loadPly(path, error);
    if (loaded.empty())
      return false;

    sorter.reset(); // joins the worker before `splats` is rewritten
    ctx->wait({});  // and drains the GPU before the buffers are replaced
    splats = std::move(loaded);
    printf("loaded %zu Gaussians from %s (%.1f MiB)\n",
           splats.size(),
           path.string().c_str(),
           double(splats.size() * sizeof(GpuSplat)) / double(1 << 20));

    const fs::path cameras = !camerasArg.empty() ? camerasArg : camerasFor(path);
    if (!cameras.empty() && g_home.loadReferenceCameras(cameras)) {
      printf("cameras: %s\n", cameras.string().c_str());
    } else {
      glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
      for (const GpuSplat& s : splats) {
        lo = glm::min(lo, glm::vec3(s.posOpacity));
        hi = glm::max(hi, glm::vec3(s.posOpacity));
      }
      const glm::vec3 center = 0.5f * (lo + hi);
      g_home = {center - glm::vec3(0, 0, glm::length(hi - lo)), center, {0, -1, 0}, {0, -1, 0}, 45.0f};
      printf(
          "no cameras.json found; framing the splats.\n"
          "  Expect a poor viewpoint -- fly in with WSAD.\n");
    }
    g_sceneScale = glm::length(g_home.eye - g_home.target);
    printf("       looking at (%.2f %.2f %.2f) from (%.2f %.2f %.2f), %.1f deg FOV\n",
           g_home.target.x,
           g_home.target.y,
           g_home.target.z,
           g_home.eye.x,
           g_home.eye.y,
           g_home.eye.z,
           g_home.fovYDegrees);

    splatBuffer = ctx->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                     .storage = lvk::StorageType_Device,
                                     .size = splats.size() * sizeof(GpuSplat),
                                     .data = splats.data(),
                                     .debugName = "Buffer: splats"});
    // One ordering buffer per frame in flight: a finished sort is uploaded
    // while the GPU may still be reading the previous one.
    for (uint32_t i = 0; i != kRingSize; i++) {
      orderBuffer[i] = ctx->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                          .storage = lvk::StorageType_HostVisible,
                                          .size = splats.size() * sizeof(uint32_t),
                                          .debugName = "Buffer: order"});
      inFlight[i] = {};
    }
    drawSlot = uploadSlot = 0;

    /*
     * GOTCHA 3: CONSTRUCT the positioner with the pose; never `lookAt` it.
     * `lookAt` does not set `up_` -- the axis the camera re-levels against on
     * every mouse move -- and `up_` has no setter. Left at its default of
     * world +Z, the first drag re-levels around the wrong axis: for a capture
     * whose up is roughly world -Y, a 156-degree roll and a black screen.
     * `cfg_` is where Space reads the pose it returns to, so it is set too.
     */
    app.cfg_.initialCameraPos = g_home.eye;
    app.cfg_.initialCameraTarget = g_home.target;
    app.cfg_.initialCameraUpVector = g_home.up;
    // Constructed with gravity, aimed with the camera's own up: the constructor is the only way to set `up_`, and
    // `lookAt` is the only way to set an orientation without disturbing it. Do it in this order and both are right --
    // dragging re-levels against gravity, while the first frame is the reference camera to the last decimal.
    app.positioner_ = CameraPositioner_FirstPerson(g_home.eye, g_home.target, g_home.gravity);
    app.positioner_.lookAt(g_home.eye, g_home.target, g_home.up);
    // Constructing resets the speeds, so rescaling belongs here. The
    // reference values are absolute, tuned for a scene ten units across, and
    // an SfM world has no absolute scale -- rescale and the dynamics carry.
    app.positioner_.maxSpeed_ *= g_sceneScale / 10.0f;
    app.positioner_.acceleration_ *= g_sceneScale / 10.0f;

    const glm::mat4 v = toOpenCvView(app.camera_.getViewMatrix());
    sorter.emplace(splats, glm::vec3(v[0][2], v[1][2], v[2][2]));
    return true;
  };

  if (!ply.empty()) {
    std::string error;
    if (!loadScene(ply, error)) {
      fprintf(stderr, "error: %s\nusage: %s [splats.ply] [cameras.json] [--screenshot out.png]\n", error.c_str(), argv[0]);
      return 1;
    }
  }

  // ---- the startup dialog --------------------------------------------------
  fs::path pendingLoad;
  std::string pickerError;

  auto drawPicker = [&]() {
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(0.5f * display.x, 0.5f * display.y), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin("Load a Gaussian splat model", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
#if !defined(NDEBUG)
    {
      // Measured on an RTX 3070 Ti: `bicycle` (6.1M splats) loads in 7.7 s
      // Release, 22.1 s Debug -- almost all of it 380 million RPly callbacks
      // into a std::vector MSVC bounds-checks. Say so, because from the
      // dialog it just looks like the viewer has hung.
      const char* line1 = "DEBUG BUILD - loading is about 3x slower than a Release build.";
      const char* line2 = "A 6-million-splat scene takes roughly 20 seconds to appear.";
      const ImGuiStyle& style = ImGui::GetStyle();
      const ImVec2 size1 = ImGui::CalcTextSize(line1), size2 = ImGui::CalcTextSize(line2);
      // Span the dialog, not just the text. The window auto-resizes to the
      // table, so its width is known only from the previous frame -- hence
      // the max(), which keeps the banner readable on the first one.
      const float width = std::max(std::max(size1.x, size2.x) + 2.0f * style.WindowPadding.x,
                                   ImGui::GetContentRegionAvail().x);
      ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.50f, 0.08f, 0.08f, 1.0f));
      ImGui::BeginChild("##debugbanner",
                        ImVec2(width, size1.y + size2.y + 2.0f * style.WindowPadding.y),
                        ImGuiChildFlags_None);
      ImGui::TextUnformatted(line1);
      ImGui::TextUnformatted(line2);
      ImGui::EndChild();
      ImGui::PopStyleColor();
      ImGui::Spacing();
    }
#endif
    if (candidates.empty()) {
      ImGui::TextUnformatted("No Gaussian splat models found near this executable.");
      ImGui::TextDisabled("Run Part 1 to produce Chapter09/work/splats.ply, or pass one on the command line.");
    } else if (ImGui::BeginTable("models", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
      ImGui::TableSetupColumn("model");
      ImGui::TableSetupColumn("splats");
      ImGui::TableSetupColumn("cameras.json");
      ImGui::TableHeadersRow();
      for (const Candidate& c : candidates) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushID(c.path.string().c_str());
        if (ImGui::Selectable(c.display.c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
          pendingLoad = c.path;
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("%s", c.path.string().c_str());
        ImGui::PopID();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(c.splatsText.c_str());
        ImGui::TableNextColumn();
        // Without one the viewer has to frame the splats itself, and for a
        // forward-facing capture that guess looks like a rendering bug.
        if (c.hasCameras)
          ImGui::TextUnformatted("yes");
        else
          ImGui::TextDisabled("no -- expect a poor first viewpoint");
      }
      ImGui::EndTable();
    }
    ImGui::Separator();
    if (!pickerError.empty())
      ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s", pickerError.c_str());
    ImGui::TextDisabled("Esc quits");
    ImGui::End();
  };

  app.run([&](ldr::Span<const RenderView> views, float) {
    // Chosen last frame. Loading waits for the GPU to go idle and replaces the
    // buffers, so it must happen outside both the ImGui frame and the command
    // buffer below.
    if (!pendingLoad.empty()) {
      const fs::path chosen = pendingLoad;
      pendingLoad.clear();
      pickerError.clear();
      loadScene(chosen, pickerError); // on failure, stay in the dialog and say why
    }

    const RenderView& view = views[0];
    const uint32_t w = uint32_t(view.viewport.width), h = uint32_t(view.viewport.height);
    const lvk::Framebuffer framebuffer = {.color = {{.texture = view.colorTexture}}};

    if (!sorter) { // nothing chosen yet: the dialog is the whole frame
      lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();
      buf.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.02f, 0.02f, 0.03f, 1.0f}}}}, framebuffer);
      app.imgui_->beginFrame(framebuffer);
      drawPicker();
      app.imgui_->endFrame(buf);
      buf.cmdEndRendering();
      ctx->submit(buf, view.colorTexture);
      return;
    }

    ensureHdr(w, h);

    const glm::vec3 eye = app.camera_.getPosition();
    const glm::mat4 viewMatrix = toOpenCvView(app.camera_.getViewMatrix());
    const glm::vec3 forward(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);

    sorter->request(forward);
    if (const std::vector<uint32_t>* fresh = sorter->take()) {
      uploadSlot = (drawSlot + 1) % kRingSize;
      ctx->wait(inFlight[uploadSlot]); // the GPU may still be reading this one
      ctx->upload(orderBuffer[uploadSlot], fresh->data(), fresh->size() * sizeof(uint32_t));
      drawSlot = uploadSlot;
    }

    const float fy = 0.5f * float(h) / std::tan(glm::radians(g_home.fovYDegrees) * 0.5f);
    const PushConstants pc{.view = viewMatrix,
                           .splats = ctx->gpuAddress(splatBuffer),
                           .order = ctx->gpuAddress(orderBuffer[drawSlot]),
                           .focal = glm::vec2(fy),
                           .viewport = glm::vec2(float(w), float(h)),
                           .cameraPos = glm::vec4(eye, 1.0f),
                           .shDegree = g_shDegree};

    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();
    buf.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.02f, 0.02f, 0.03f, 1.0f}}}},
                          {.color = {{.texture = hdr}}});
    buf.cmdBindRenderPipeline(pipeline);
    // No depth test: correctness comes from the sort, and a depth test would
    // reject the semi-transparent splats that must blend through.
    buf.cmdBindDepthState({});
    buf.cmdPushConstants(pc);
    buf.cmdDraw(6, uint32_t(splats.size()));
    buf.cmdEndRendering();

    buf.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_DontCare}}}, framebuffer, {.sampledImages = {hdr}});
    buf.cmdBindRenderPipeline(resolve);
    buf.cmdBindDepthState({});
    buf.cmdPushConstants(ResolveConstants{hdr.index(), sampler.index(), glm::vec2(float(w), float(h))});
    buf.cmdDraw(3);

    // Overlay only when a human is watching: a screenshot is the scene alone,
    // which 02_compare_renderers.py differences pixel for pixel.
    if (!app.cfg_.screenshotFrameNumber) {
      app.imgui_->beginFrame(framebuffer);
      ImGui::SetNextWindowPos({0, 0});
      ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
      ImGui::Text("W/S/A/D - camera movement");
      ImGui::Text("1/2 - camera up/down");
      ImGui::Text("Shift - fast movement");
      ImGui::Text("Left mouse - look around");
      ImGui::Text("Wheel - dolly");
      ImGui::Text("Space - reset camera");
      ImGui::Text("[ / ] - spherical harmonics: degree %d", g_shDegree);
      ImGui::End();
      app.drawFPS();
      app.imgui_->endFrame(buf);
    }
    buf.cmdEndRendering();

    inFlight[drawSlot] = ctx->submit(buf, view.colorTexture);
  });

  VULKAN_APP_EXIT();
}
