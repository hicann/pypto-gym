// ascriptor pto_isa backend support header (a5 / dav-c310).
//
// Deliberately tiny, and the boundary is the point. `tensorutils_cce.h` is 1800 lines because CCE
// builtins are raw -- `copy_gm_to_ubuf(dst, src, sid, nBurst, lenBurst, srcStride, dstStride)` has
// no types and no shapes -- so that header has to supply the whole machine view: Tensor, GMTensor,
// Buff, Event, and one wrapper per instruction.
//
// PTO already *is* that layer. `TLOAD(tile, gtensor)` carries the shape in the tile's type, which
// is the premise of RFC-0011 and the reason an audit of `tensorutils_cce.h`'s 114 names found
// only two that this backend needed. So there is deliberately **no wrapper here over any T*
// instruction and none over `pto::Tile`**: every emitted line stays checkable directly against
// PTO's own headers, which is how §4.3's fractal mapping and §4.4's `srcStride = TileData::Rows`
// were settled without a board, and the tile's template arguments *are* the shape information
// this backend refuses on when it does not fold.
//
// What is here is only what PTO does not provide and our own lowering decides:
//
//   Min / Max /      the scalar helpers the shared scalar printer emits (§7.5), and
//   CeilDiv / AlignUp  that a @vf body reaches for (§6)
//   deq_scalar       the fixpipe quant SPR word -- PTO picks the QuantMode_t itself but takes
//                    the word as an argument, and it is cce's word (§4.11)
//   Flags / Event    a depth>1 event's id rotation and the counters that drive it, over ids
//                    the events pass has already assigned
//   SlotOf           a slot buffer's ring index -- `Buff::get`'s arithmetic, without the Buff
//   CUBE_READY, ...  the cross-core flag pairs, whose N / N+16 asymmetry is a hardware fact
//
// The cross-core block and Min / Max are restated **byte for byte** from `tensorutils_cce.h` so
// that the two backends' output is diffable line for line rather than only semantically (which is
// what `tools/pto_isa_check_xcore.py` had to do while these were printed inline). Only the a5 /
// c310 branch is restated: this backend is a5-only (RFC-0011 §1).
//
// Include order: after `kernel_operator.h`, which supplies pipe_t / event_t / set_flag and the
// cross-core intrinsics; the pto_isa backend's HEADER emits them in that order.
#pragma once

#include <stdint.h>
#include "scalar_math.h"

namespace ascrip {

// The scalar layer (RFC-0011 §7.5) is the cce printer's, unmodified: `scalar.min` prints `Min`,
// `scalar.ceil_div` prints `CeilDiv`, `scalar.align` prints `AlignUp`. cce lets
// `tensorutils_cce.h:225-230` define all four, so this header restates them byte for byte for the
// same reason the cross-core block is restated -- the emitted text stays identical, and the
// checkers that diff the two backends stay textual. `pto::ReduceOp::Min` is a scoped enum, so
// `using namespace pto` puts no competing `Min` in scope.
__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b) { return b == 0 ? 0 : (a + b - 1) / b; }
__aicore__ inline int32_t AlignUp(int32_t a, int32_t n) { return n == 0 ? 0 : (a + n - 1) / n * n; }
template <typename A, typename B>
__aicore__ inline A Min(A a, B b) { return (a < b) ? a : (A)b; }
template <typename A, typename B>
__aicore__ inline A Max(A a, B b) { return (a < b) ? (A)b : a; }

// A gmlist parameter: AscendC's ListTensorDesc in its DYNAMIC layout (RFC-0001 §13), read with
// plain scalar loads. Restated byte for byte from `tensorutils_cce.h:314` -- and it is here rather
// than refused because there is no instruction in it: every member is a `__gm__ uint64_t` load off
// the descriptor, PTO has nothing to say about descriptors, and what the reads *produce* is a
// pointer that §4.1's `pto::GlobalTensor` then takes like any other. head[0] is the byte offset of
// the pointer array; from head[1] every member has a header whose low half is the rank (the high
// half is a framework word: 1 on the board and cannsim, not the index) followed by its rank dims;
// the pointer array holds one GM address per member. count() is what the layout implies -- the
// pointer array offset divided by the per-member descriptor size (CANN's own consistency rule) --
// and so does not depend on that word.
template <typename T>
struct GMList {
    __gm__ uint64_t* head;
    __aicore__ inline GMList(__gm__ uint8_t* p) : head((__gm__ uint64_t*)p) {}
    __aicore__ inline uint32_t rank() const { return (uint32_t)(head[1] & 0xffffffffu); }
    __aicore__ inline int32_t count() const
    {
        uint32_t desc = rank() ? 1 + rank() : 2;
        return (int32_t)((head[0] - 8) / (desc * 8));
    }
    __aicore__ inline __gm__ T* ptr(int32_t i) const { return (__gm__ T*)head[head[0] / 8 + i]; }
    __aicore__ inline int64_t dim(int32_t i, int32_t d) const { return (int64_t)head[1 + i * (1 + rank()) + 1 + d]; }
};

// The fixpipe scalar-quant SPR word (§4.11). PTO's quantised TSTORE / TMOV / TEXTRACT overloads
// pick the `QuantMode_t` themselves -- `GetScalarPreQuantMode<Src, Dst>()` is a constexpr over the
// tile dtypes, the same pair cce's `fixpipe_quant` branches on -- but they take the SPR word as a
// plain `uint64_t` argument and write it with the same `set_quant_pre`. So the mode agrees by
// construction and only the word is ours to build; this is `tensorutils_cce.h`'s c310
// `pack_deq_scalar`, restated byte for byte, so the two backends' quant SPR can be diffed rather
// than argued about: [31:13] the fp32 scale's top bits, [45:37] a 9-bit offset, [46] signed
// saturation.
__aicore__ inline uint64_t pack_deq_scalar(float scale, int32_t offset)
{
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = scale;
    uint64_t deq = (uint64_t)(cvt.u & 0xFFFFE000u);
    deq |= ((uint64_t)((uint32_t)offset & 0x1FFu)) << 37;
    return deq;
}

// The two bits cce derives from the dtype pair, passed in instead of derived: only its two 8-bit
// arms (QF322B8_PRE, REQ8) carry the offset -- every other arm packs `pack_deq_scalar(scale, 0)`
// and drops it -- and only a *signed* 8-bit destination sets bit 46. The printer knows the pair
// (it has to: it checks the pair against PTO's own table first), so deciding there keeps this file
// free of the AscendC type traits `fixpipe_quant` uses, which the PTO translation unit does not
// have.
template <bool WithOffset, bool SignedB8>
__aicore__ inline uint64_t deq_scalar(float scale, int32_t offset)
{
    uint64_t deq = pack_deq_scalar(scale, WithOffset ? offset : 0);
    if constexpr (SignedB8) {
        deq |= (1ULL << 46);
    }
    return deq;
}

// A slot buffer's ring index. cce spells a slot buffer as an *object* and applies the index by
// calling `Buff::get(i)`, which wraps: `slot[((i % N) + N) % N]` (tensorutils_cce.h:375). This
// backend spells the same buffer as a C++ array of tiles one aligned slot apart, so the index is
// also address arithmetic (`base + SlotOf<N>(i) * step`) and there is no Buff to hang it on --
// but the wrap is the same, and it is load-bearing: our kernels drive a ring with a counter that
// only ever increments, so an unwrapped index runs off the end of the array within two
// iterations. It was inlined at 2384 call sites before this header existed, which is 2384 chances
// to write it differently.
template <int N>
__aicore__ inline int SlotOf(int i) { return ((i % N) + N) % N; }

// ===========================================================================================
// Cross-core synchronisation (sync.crosscore.*). a5/C310: cube <-> vector point-to-point rides
// mode 0x4 (intra-block); the AIC side sets / waits BOTH flag N and N+16 because AIV1's N is
// remapped to N+16, the AIV side handles the single N. The ALL* / INTRACORE groups stay on the
// FFTS path (mode 0x0 / 0x1); message packing from dav_3510 kernel_operator_sync_impl.h.
//
// PTO does have cross-core events -- `pto::Event` with `IsCrossCore`, whose `Init` is exactly
// `set_intra_block(p, id); set_intra_block(p, id + 16);` and whose `Wait` is a single
// `wait_intra_block` (npu/a5/TSync.hpp:98-116), the same two intrinsics -- but it ties the event
// to a data-movement op pair (`IsCrossCoreEvent()` is true only for TMOV_A2V / TMOV_V2M /
// TEXTRACT_V2M) while our IR's cross-core flags are free-standing, with ids the autosync pass has
// already assigned. Same reason §5 declines PTO's own event class for the intra-core flags.
//
// The `if ASCEND_IS_AIC` / `AIV` guards live in the helper, not at the call site: a vector-only
// intrinsic is a compile error in the cube translation unit and vice versa, and a constexpr-if
// the unit never odr-uses compiles to nothing (the compile-unit model at the top of
// tensorutils_cce.h, board-verified).
// ===========================================================================================
template <int MODE, pipe_t P>
__aicore__ inline void CrossCoreSetFlag(uint16_t flag_id)
{
    if constexpr (MODE == 0x4) {
        set_intra_block(P, flag_id);
    } else {
        ffts_cross_core_sync(P, 0x1ull | ((uint64_t)(MODE & 0x3) << 4) | ((uint64_t)(flag_id & 0xf) << 8));
    }
}
template <int MODE, pipe_t P>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flag_id)
{
    if constexpr (MODE == 0x4) {
        wait_intra_block(P, flag_id);
    } else {
        wait_flag_dev(P, flag_id);
    }
}
template <pipe_t P>
__aicore__ inline void CUBE_READY(int id)
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0x4, P>((uint16_t)id);
        CrossCoreSetFlag<0x4, P>((uint16_t)(id + 16));
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_VEC(int id)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<0x4, P>((uint16_t)id);
        CrossCoreWaitFlag<0x4, P>((uint16_t)(id + 16));
    }
}
template <pipe_t P>
__aicore__ inline void VEC_READY(int id)
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0x4, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_CUBE(int id)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<0x4, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void ALLCUBE_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLCUBE_WAIT(int id) { CrossCoreWaitFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_WAIT(int id) { CrossCoreWaitFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_READY(int id) { CrossCoreSetFlag<0x1, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_WAIT(int id) { CrossCoreWaitFlag<0x1, P>((uint16_t)id); }

// A depth>1 event's tokens rotate through its flag ids, and which id a given set uses is a
// run-time question (`set_cnt % depth`) while `set_flag`'s id operand is a compile-time one. The
// bridge is an if-chain over the ids. Restated byte for byte from `tensorutils_cce.h:388-455`,
// including the `Event` that owns the rotation counters: PRESET tokens are set in the constructor
// and drained in the destructor, the SEvent/DEvent/TEvent/QEvent protocol at any depth.
//
// Nothing here allocates -- the ids are template arguments, chosen by the events pass -- which is
// why it avoids what RFC-0011 §5 refuses PTO's own `Event` class for (`EventIdCounter` picks ids
// itself and would fight the pass that owns them).
//
// The destructor is the load-bearing part. `release()` waits, at scope exit, every set that was
// never waited, counting at run time; that is correct at *every* return, including one inside a
// loop or a branch. Printing the bare pair instead meant deciding statically how many flags were
// outstanding, which needs a set-minus-wait analysis over the whole function, a run-time counter
// fallback where a `cf.if`'s arms disagree, and a refusal where an early return sits inside a
// region. C++ scoping does all three for free.
template <pipe_t SET, pipe_t WAIT, int... IDS>
struct Flags {
    static constexpr int depth = sizeof...(IDS);
    template <int ID, int... REST>
    __aicore__ static inline void set_nth(int k)
    {
        if (k == 0) {
            set_flag(SET, WAIT, (event_t)ID);
        } else if constexpr (sizeof...(REST) > 0) {
            set_nth<REST...>(k - 1);
        }
    }
    template <int ID, int... REST>
    __aicore__ static inline void wait_nth(int k)
    {
        if (k == 0) {
            wait_flag(SET, WAIT, (event_t)ID);
        } else if constexpr (sizeof...(REST) > 0) {
            wait_nth<REST...>(k - 1);
        }
    }
    __aicore__ static inline void set(int k) { set_nth<IDS...>(k); }
    __aicore__ static inline void wait(int k) { wait_nth<IDS...>(k); }
};

template <pipe_t SET, pipe_t WAIT, int PRESET, int... IDS>
class Event {
public:
    static constexpr int depth = sizeof...(IDS);
    __aicore__ inline Event()
    {
        for (int i = 0; i < PRESET; ++i) {
            set();
        }
    }
    __aicore__ inline ~Event() { release(); }
    __aicore__ inline void set()
    {
        Flags<SET, WAIT, IDS...>::set(set_cnt % depth);
        set_cnt += 1;
    }
    __aicore__ inline void wait()
    {
        Flags<SET, WAIT, IDS...>::wait(wait_cnt % depth);
        wait_cnt += 1;
    }
    __aicore__ inline void set_all()
    {
        for (int i = 0; i < depth; ++i) {
            set();
        }
    }
    __aicore__ inline void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    int set_cnt = 0;
    int wait_cnt = 0;
};
// The old event names by depth (ids appended, as the events pass assigns them).
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1>
using SEvent = Event<SET, WAIT, PRESET, ID1>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2>
using DEvent = Event<SET, WAIT, PRESET, ID1, ID2>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2, int ID3>
using TEvent = Event<SET, WAIT, PRESET, ID1, ID2, ID3>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2, int ID3, int ID4>
using QEvent = Event<SET, WAIT, PRESET, ID1, ID2, ID3, ID4>;


// ============================================================================================
// `is_same` and the SIMT shim, restated from `tensorutils_cce.h:260, 2021` byte for byte -- the
// same boundary the cross-core block and `GMList` above are on, and for the same reason (§7.13).
//
// A `@simt` body is plain C on the *compiler's* SIMT layer (`__clang_cce_simt*.h`: threadIdx /
// blockDim / the atomic family / __sync_workitems), over `__gm__` and `__ubuf__` pointers. There
// is no tile in it and nothing for PTO to say about it, which is precisely the argument §7.7
// makes for printing `@vf` bodies with cce's own `VfPrinter` instead of reprinting them here: a
// second copy of the same C is a second chance to be wrong. So this backend prints the *launch*
// and the body comes from cce's `SimtPrinter`, and the shim they both call has to be here.
// ============================================================================================
template <typename A, typename B>
struct is_same { static constexpr bool value = false; };
template <typename A>
struct is_same<A, A> { static constexpr bool value = true; };

// SIMT shim: the launch and the few helpers a printed @simt function calls, on the compiler's own
// SIMT layer (__clang_cce_simt*.h: threadIdx / blockDim / blockIdx / gridDim, the atomic family,
// __sync_workitems, cce::dim3 / cce::async_invoke). Semantics transcribed from CANN 9.2
// asc/impl/simt_api/cpp/dav_3510 (fire-and-forget plus readback where the hardware op returns
// nothing).
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // simt shim: rides c310's own SIMT layer
namespace simt {
template <auto FN, typename... Args>
__aicore__ inline void launch(int threads, Args&&... args)
{
#if defined(__DAV_VEC__)
    cce::async_invoke<FN>(cce::dim3{(unsigned)threads, 1u, 1u}, args...);
#else
    (void)threads;
#endif
}
__simt_callee__ inline uint32_t thread_id() { return threadIdx.x; }
__simt_callee__ inline uint32_t thread_num() { return blockDim.x; }
__simt_callee__ inline uint32_t blk_idx() { return blockIdx.x; }
__simt_callee__ inline uint32_t blk_num() { return gridDim.x; }
__simt_callee__ inline void barrier() { __sync_workitems(); }

template <typename T>
struct returns_old_ub {
    static constexpr bool value = is_same<T, int32_t>::value || is_same<T, uint32_t>::value || is_same<T, float>::value;
};
template <typename T>
struct returns_old_gm {
    static constexpr bool value = returns_old_ub<T>::value || is_same<T, int64_t>::value || is_same<T, uint64_t>::value;
};

#define ASCRIP_SIMT_ATOMIC(NAME, INTRINSIC, SPACE, TRAIT)                             \
    template <typename T>                                                          \
    __simt_callee__ inline T NAME(SPACE T* address, T val)                        \
    {                                                                              \
        if constexpr (TRAIT<T>::value) {                                           \
            return INTRINSIC(address, val);                                        \
        } else {                                                                   \
            INTRINSIC(address, val);                                               \
            return *address;                                                       \
        }                                                                          \
    }
ASCRIP_SIMT_ATOMIC(atomic_add, atomicAdd, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_add, atomicAdd, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_max, atomicMax, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_max, atomicMax, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_min, atomicMin, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_min, atomicMin, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_exch, atomicExch, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_exch, atomicExch, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_and, atomicAnd, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_and, atomicAnd, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_or, atomicOr, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_or, atomicOr, __gm__, returns_old_gm)
ASCRIP_SIMT_ATOMIC(atomic_xor, atomicXOr, __ubuf__, returns_old_ub)
ASCRIP_SIMT_ATOMIC(atomic_xor, atomicXOr, __gm__, returns_old_gm)
#undef ASCRIP_SIMT_ATOMIC
template <typename T>
__simt_callee__ inline T atomic_sub(__ubuf__ T* address, T val)
{
    if constexpr (returns_old_ub<T>::value) {
        return atomicSub(address, val);
    } else {
        atomicAdd(address, -val);
        return *address;
    }
}
template <typename T>
__simt_callee__ inline T atomic_sub(__gm__ T* address, T val)
{
    if constexpr (returns_old_gm<T>::value) {
        return atomicSub(address, val);
    } else {
        atomicAdd(address, -val);
        return *address;
    }
}
template <typename T>
__simt_callee__ inline T atomic_cas(__ubuf__ T* address, T compare, T val) { return atomicCAS(address, compare, val); }
template <typename T>
__simt_callee__ inline T atomic_cas(__gm__ T* address, T compare, T val) { return atomicCAS(address, compare, val); }

// ring inc/dec (CUDA semantics: inc wraps past `limit` to 0, dec wraps 0 / >limit to `limit`)
__simt_callee__ inline uint32_t atomic_inc(__ubuf__ uint32_t* address, uint32_t limit) { return atomicInc(address, limit); }
__simt_callee__ inline uint32_t atomic_inc(__gm__ uint32_t* address, uint32_t limit) { return atomicInc(address, limit); }
__simt_callee__ inline uint32_t atomic_dec(__ubuf__ uint32_t* address, uint32_t limit) { return atomicDec(address, limit); }
__simt_callee__ inline uint32_t atomic_dec(__gm__ uint32_t* address, uint32_t limit) { return atomicDec(address, limit); }

__simt_callee__ inline void threadfence() { __threadfence(); }
__simt_callee__ inline void threadfence_block() { __threadfence_block(); }

// ---- scalar math on the SIMT layer. The dav-c310 compiler builtins are the same layer the
// PyPTO Pro CCE codegen prints (__expf / __logf / __sqrtf / __fma / ...); everything else is
// synthesised here the way its backend and the CANN dav_c310 SIMT impl do.
__simt_callee__ inline float exp(float x) { return __expf(x); }
__simt_callee__ inline float log(float x) { return __logf(x); }
__simt_callee__ inline float exp2(float x) { return __expf(x * 0.6931471805599453f); }
__simt_callee__ inline float log2(float x) { return __logf(x) * 1.4426950408889634f; }
__simt_callee__ inline float log1p(float x) { return __logf(1.0f + x); }
__simt_callee__ inline float rsqrt(float x) { return 1.0f / __sqrtf(x); }
__simt_callee__ inline float tanh(float x) { return 1.0f - (2.0f / (__expf(2.0f * x) + 1.0f)); }
// A5 native integral rounding can canonicalize -0 to +0 (M10-054). Adapt that
// target detail here, using integer fields so signed-zero/fast-math rewrites
// cannot turn the zero-magnitude test or restored sign into floating arithmetic.
__simt_callee__ inline float integral_zero_sign_(float x, float result)
{
    union { float f; uint32_t u; } source, rounded;
    source.f = x;
    rounded.f = result;
    if ((rounded.u & 0x7FFFFFFFu) == 0u)
        rounded.u = source.u & 0x80000000u;
    return rounded.f;  // Every nonzero result, including NaN payloads, is unchanged.
}
__simt_callee__ inline float trunc_native_(float x) { return x >= 0.0f ? __floorf(x) : __ceilf(x); }
__simt_callee__ inline float rint(float x) { return integral_zero_sign_(x, __rintf(x)); }
__simt_callee__ inline float round(float x) { return integral_zero_sign_(x, __roundf(x)); }
__simt_callee__ inline float floor(float x) { return integral_zero_sign_(x, __floorf(x)); }
__simt_callee__ inline float ceil(float x) { return integral_zero_sign_(x, __ceilf(x)); }
__simt_callee__ inline float trunc(float x) { return integral_zero_sign_(x, trunc_native_(x)); }
// Exact FP32 remainder (RFC-0001 §6.15), as PyPTO Pro prints it: long division of the
// integer significand fields never rounds, keeps subnormals and gives a zero result the
// dividend's sign. A NaN operand, infinite dividend or zero divisor is quiet NaN 0x7FC00000.
__simt_callee__ inline float fmod(float x, float y)
{
    union { float f; uint32_t u; } dividend, divisor;
    dividend.f = x;
    divisor.f = y;
    uint32_t sign = dividend.u & 0x80000000u;
    uint32_t ax = dividend.u & 0x7FFFFFFFu, ay = divisor.u & 0x7FFFFFFFu;
    if (ay == 0u || ax >= 0x7F800000u || ay > 0x7F800000u) {
        dividend.u = 0x7FC00000u;
    } else if (ax >= ay) {  // Otherwise |x| < |y| and x is the remainder.
        int32_t ex = (int32_t)(ax >> 23), ey = (int32_t)(ay >> 23);
        uint32_t mx = ax & 0x007FFFFFu, my = ay & 0x007FFFFFu;
        if (ex == 0) {
            ex = 1;
            while (mx < 0x00800000u) { mx <<= 1; --ex; }
        } else {
            mx |= 0x00800000u;
        }
        if (ey == 0) {
            ey = 1;
            while (my < 0x00800000u) { my <<= 1; --ey; }
        } else {
            my |= 0x00800000u;
        }
        while (ex > ey) {
            if (mx >= my) mx -= my;
            mx <<= 1;
            --ex;
        }
        if (mx >= my) mx -= my;
        if (mx == 0u) {
            dividend.u = sign;
        } else {
            while (mx < 0x00800000u) { mx <<= 1; --ex; }
            dividend.u = sign | (ex > 0 ? (mx & 0x007FFFFFu) | ((uint32_t)ex << 23) : mx >> (1 - ex));
        }
    }
    return dividend.f;
}
__simt_callee__ inline int32_t ffs(uint32_t x)  // 1-based index of the lowest set bit; 0 when empty
{
    return x ? (int32_t)__popc((x & (0u - x)) - 1u) + 1 : 0;
}

// sin / cos: the CANN dav_c310 SIMT reduction + polynomials (Cody-Waite for small angles,
// Payne-Hanek for |x| > 71476.0625f), the same pair the PyPTO Pro codegen inlines.
__simt_callee__ inline float sincos_reduce_ph_(float x, int32_t* quadrant)
{
    uint32_t bits = reinterpret_cast<uint32_t&>(x);
    int32_t exponent = (int32_t)((bits & 0x7F800000u) >> 23) - 127;
    uint32_t ei = (uint32_t)exponent >> 5;
    const uint32_t tbl[] = {0x517cc1b7u, 0x27220a94u, 0xfe13abe8u, 0xfa9a6ee0u, 0x6db14accu, 0x9e21c820u};
    uint32_t hi = ei ? tbl[ei - 1] : 0u;
    uint32_t mid = tbl[ei];
    uint32_t lo = tbl[ei + 1];
    uint32_t last = tbl[ei + 2];
    int32_t rem = (int32_t)((uint32_t)exponent & 0x1Fu);
    if (rem != 0) {
        hi = (hi << rem) | (mid >> (32 - rem));
        mid = (mid << rem) | (lo >> (32 - rem));
        lo = (lo << rem) | (last >> (32 - rem));
    }
    uint32_t mant = (bits & 0x007FFFFFu) | 0x4F000000u;
    uint32_t nm = (uint32_t)reinterpret_cast<float&>(mant);
    uint64_t prod = (uint64_t)nm * lo;
    prod = (uint64_t)nm * mid + (prod >> 32);
    prod = ((uint64_t)(nm * hi) << 32) + prod;
    int32_t q = (int32_t)(prod >> 62);
    prod &= 0x3FFFFFFFFFFFFFFFull;
    if (prod & 0x2000000000000000ull) {
        prod -= 0x4000000000000000ull;
        q += 1;
    }
    int64_t p64 = (int64_t)prod;
    float hf = (float)p64;
    p64 -= (int64_t)hf;
    float lf = (float)p64;
    float reduced = (hf + lf) * 3.4061215800865545e-19f;  // pi/2 * 2^-62
    if (x < 0.0f) {
        reduced = -reduced;
        q = -q;
    }
    *quadrant = q;
    return reduced;
}
__simt_callee__ inline float sincos_reduce_(float x, int32_t* quadrant)
{
    x = __fma(x, 0.0f, x);
    if (__fabsf(x) > 71476.0625f) {
        return sincos_reduce_ph_(x, quadrant);
    }
    float y = __fma(x, 0.636619747f, 12582912.0f);  // 2/pi; 1.5*2^23 truncates the mantissa
    *quadrant = reinterpret_cast<int32_t&>(y);
    y = y - 12582912.0f;
    x = __fma(y, -1.57079601e+00f, x);
    x = __fma(y, -3.13916473e-07f, x);
    return __fma(y, -5.39030253e-15f, x);
}
__simt_callee__ inline float sincos_cospoly_(float x)
{
    x = x * x;
    float y = __fma(x, 2.44677067e-5f, -1.38877297e-3f);
    y = __fma(x, y, 4.16666567e-2f);
    y = __fma(x, y, -5.00000000e-1f);
    return __fma(x, y, 1.00000000e+0f);
}
__simt_callee__ inline float sincos_sinpoly_(float x)
{
    float y = x * x;
    float m = __fma(x, y, 0.0f);
    float z = __fma(y, 2.86567956e-6f, -1.98559923e-4f);
    z = __fma(y, z, 8.33338592e-3f);
    z = __fma(y, z, -1.66666672e-1f);
    return __fma(z, m, x);
}
__simt_callee__ inline float sin(float x)
{
    int32_t q;
    float y = sincos_reduce_(x, &q);
    float c = sincos_cospoly_(y);
    float s = sincos_sinpoly_(y);
    if (q & 2) { s = -s; c = -c; }
    if (q & 1) { s = c; }
    return s;
}
__simt_callee__ inline float cos(float x)
{
    int32_t q;
    float y = sincos_reduce_(x, &q);
    float c = sincos_cospoly_(y);
    float s = sincos_sinpoly_(y);
    if (q & 2) { s = -s; c = -c; }
    if (q & 1) { c = -s; }
    return c;
}
}  // namespace simt
#endif  // simt

}  // namespace ascrip
