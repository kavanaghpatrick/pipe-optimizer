#!/usr/bin/env python3
"""
Python ctypes wrapper for libforge_ffi.dylib (Metal GPU compute).

Provides RAII classes for forge-ffi C-ABI:
  ForgeContext  - GPU context lifecycle
  ForgeBuffer   - Zero-copy UMA buffer management
  ForgeKernel   - Runtime Metal kernel compilation
  ForgePipeline - Pipeline builder with dispatch + execute
  ForgeError    - Exception type for FFI failures
  ForgeDtype    - Data type enum matching C ForgeDtype
"""

import ctypes
import os
import sys
from enum import IntEnum

import numpy as np


class ForgeError(Exception):
    """Raised when a forge-ffi call fails."""
    pass


class ForgeDtype(IntEnum):
    U32 = 0
    I32 = 1
    F32 = 2
    U64 = 3
    I64 = 4
    F16 = 5


# Map string aliases to ForgeDtype
_DTYPE_ALIASES = {
    'u32': ForgeDtype.U32,
    'i32': ForgeDtype.I32,
    'f32': ForgeDtype.F32,
    'u64': ForgeDtype.U64,
    'i64': ForgeDtype.I64,
    'f16': ForgeDtype.F16,
}

# Map numpy dtypes to ForgeDtype
_NUMPY_TO_FORGE = {
    np.dtype(np.uint32): ForgeDtype.U32,
    np.dtype(np.int32): ForgeDtype.I32,
    np.dtype(np.float32): ForgeDtype.F32,
    np.dtype(np.uint64): ForgeDtype.U64,
    np.dtype(np.int64): ForgeDtype.I64,
    np.dtype(np.float16): ForgeDtype.F16,
}

# Map ForgeDtype to numpy dtype
_FORGE_TO_NUMPY = {v: k for k, v in _NUMPY_TO_FORGE.items()}


def _resolve_dtype(dtype):
    """Convert string or numpy dtype to ForgeDtype."""
    if isinstance(dtype, ForgeDtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        raise ValueError(f"Unknown dtype string: {dtype!r}")
    if isinstance(dtype, np.dtype):
        if dtype in _NUMPY_TO_FORGE:
            return _NUMPY_TO_FORGE[dtype]
        raise ValueError(f"Unsupported numpy dtype: {dtype}")
    raise TypeError(f"Expected ForgeDtype, str, or np.dtype, got {type(dtype)}")


def _find_dylib():
    """Locate libforge_ffi.dylib by searching common paths."""
    name = "libforge_ffi.dylib"
    search_paths = []

    # 1. Current directory
    search_paths.append(os.path.abspath("."))

    # 2. Directory containing this file
    search_paths.append(os.path.dirname(os.path.abspath(__file__)))

    # 3. PyInstaller _MEIPASS
    if hasattr(sys, "_MEIPASS"):
        search_paths.append(sys._MEIPASS)

    # 4. Known build location
    home = os.path.expanduser("~")
    search_paths.append(
        os.path.join(home, "gpu_kernel", "metal-forge-compute", "target", "release")
    )

    # 5. DYLD_LIBRARY_PATH
    dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if dyld:
        search_paths.extend(dyld.split(":"))

    for d in search_paths:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path

    raise OSError(
        f"Cannot find {name}. Searched: {search_paths}. "
        "Build with: cd ~/gpu_kernel/metal-forge-compute && cargo build -p forge-ffi --release"
    )


def _load_lib():
    """Load the dylib and set up argtypes/restypes."""
    path = _find_dylib()
    lib = ctypes.CDLL(path)

    # Opaque pointer types
    c_ctx_p = ctypes.c_void_p
    c_buf_p = ctypes.c_void_p
    c_pip_p = ctypes.c_void_p
    c_kern_p = ctypes.c_void_p

    # -- Version & error --
    lib.forge_version.argtypes = []
    lib.forge_version.restype = ctypes.c_char_p

    lib.forge_last_error_code.argtypes = []
    lib.forge_last_error_code.restype = ctypes.c_int32

    lib.forge_last_error_message.argtypes = []
    lib.forge_last_error_message.restype = ctypes.c_char_p

    # -- Context --
    lib.forge_context_create.argtypes = []
    lib.forge_context_create.restype = c_ctx_p

    lib.forge_context_destroy.argtypes = [c_ctx_p]
    lib.forge_context_destroy.restype = None

    # -- Buffer --
    lib.forge_buffer_alloc.argtypes = [c_ctx_p, ctypes.c_size_t, ctypes.c_int]
    lib.forge_buffer_alloc.restype = c_buf_p

    lib.forge_buffer_from_data.argtypes = [
        c_ctx_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.forge_buffer_from_data.restype = c_buf_p

    lib.forge_buffer_contents.argtypes = [c_buf_p]
    lib.forge_buffer_contents.restype = ctypes.c_void_p

    lib.forge_buffer_len.argtypes = [c_buf_p]
    lib.forge_buffer_len.restype = ctypes.c_size_t

    lib.forge_buffer_byte_len.argtypes = [c_buf_p]
    lib.forge_buffer_byte_len.restype = ctypes.c_size_t

    lib.forge_buffer_dtype.argtypes = [c_buf_p]
    lib.forge_buffer_dtype.restype = ctypes.c_int

    lib.forge_buffer_destroy.argtypes = [c_buf_p]
    lib.forge_buffer_destroy.restype = None

    # -- Pipeline --
    lib.forge_pipeline_create.argtypes = [c_ctx_p]
    lib.forge_pipeline_create.restype = c_pip_p

    lib.forge_pipeline_execute.argtypes = [c_pip_p]
    lib.forge_pipeline_execute.restype = ctypes.c_int32

    lib.forge_pipeline_execute_timed.argtypes = [c_pip_p, ctypes.POINTER(ctypes.c_double)]
    lib.forge_pipeline_execute_timed.restype = ctypes.c_int32

    lib.forge_pipeline_destroy.argtypes = [c_pip_p]
    lib.forge_pipeline_destroy.restype = None

    # -- Kernel --
    lib.forge_kernel_compile.argtypes = [c_ctx_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.forge_kernel_compile.restype = c_kern_p

    lib.forge_pipeline_dispatch_1d.argtypes = [
        c_pip_p, c_kern_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
        ctypes.c_uint64,
    ]
    lib.forge_pipeline_dispatch_1d.restype = ctypes.c_int32

    lib.forge_kernel_destroy.argtypes = [c_kern_p]
    lib.forge_kernel_destroy.restype = None

    # -- Pipeline zero --
    lib.forge_pipeline_zero.argtypes = [c_pip_p, c_buf_p]
    lib.forge_pipeline_zero.restype = ctypes.c_int32

    return lib


# Module-level singleton
_lib = _load_lib()


def _check_error(msg="forge-ffi call failed"):
    """Check thread-local error state and raise ForgeError if set."""
    code = _lib.forge_last_error_code()
    if code != 0:
        err_msg = _lib.forge_last_error_message()
        if err_msg:
            err_msg = err_msg.decode("utf-8", errors="replace")
        else:
            err_msg = "unknown error"
        raise ForgeError(f"{msg}: [{code}] {err_msg}")


class ForgeContext:
    """RAII wrapper for forge_ctx_t. Supports context manager protocol."""

    def __init__(self):
        self._ptr = _lib.forge_context_create()
        if not self._ptr:
            _check_error("Failed to create GPU context")
            raise ForgeError("forge_context_create returned NULL")

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._ptr:
            _lib.forge_context_destroy(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    @property
    def ptr(self):
        if not self._ptr:
            raise ForgeError("Context already destroyed")
        return self._ptr


class ForgeBuffer:
    """RAII wrapper for forge_buf_t with zero-copy UMA access."""

    def __init__(self, ptr, dtype):
        self._ptr = ptr
        self._dtype = dtype

    @classmethod
    def from_numpy(cls, ctx, arr):
        """Create a GPU buffer from a numpy array (copies data)."""
        arr = np.ascontiguousarray(arr)
        dtype = _resolve_dtype(arr.dtype)
        ptr = _lib.forge_buffer_from_data(
            ctx.ptr,
            arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(arr.size),
            ctypes.c_int(int(dtype)),
        )
        if not ptr:
            _check_error("Failed to create buffer from numpy array")
            raise ForgeError("forge_buffer_from_data returned NULL")
        return cls(ptr, dtype)

    @classmethod
    def alloc(cls, ctx, count, dtype='f32'):
        """Allocate a zero-initialized GPU buffer."""
        dtype = _resolve_dtype(dtype)
        ptr = _lib.forge_buffer_alloc(
            ctx.ptr,
            ctypes.c_size_t(count),
            ctypes.c_int(int(dtype)),
        )
        if not ptr:
            _check_error(f"Failed to allocate buffer ({count} elements, {dtype.name})")
            raise ForgeError("forge_buffer_alloc returned NULL")
        return cls(ptr, dtype)

    def to_numpy(self):
        """Zero-copy view of buffer contents as numpy array.

        The returned array shares memory with the GPU buffer via UMA.
        Valid only while this ForgeBuffer is alive.
        """
        if not self._ptr:
            raise ForgeError("Buffer already destroyed")
        contents = _lib.forge_buffer_contents(self._ptr)
        if not contents:
            raise ForgeError("forge_buffer_contents returned NULL")
        length = _lib.forge_buffer_len(self._ptr)
        np_dtype = _FORGE_TO_NUMPY[self._dtype]
        arr = np.ctypeslib.as_array(
            (ctypes.c_uint8 * (length * np_dtype.itemsize)).from_address(contents)
        )
        return arr.view(np_dtype)[:length].copy()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._ptr:
            _lib.forge_buffer_destroy(self._ptr)
            self._ptr = None

    @property
    def ptr(self):
        if not self._ptr:
            raise ForgeError("Buffer already destroyed")
        return self._ptr


class ForgeKernel:
    """RAII wrapper for a compiled Metal kernel."""

    def __init__(self, ctx, msl_source, fn_name):
        """Compile a Metal kernel from source.

        Args:
            ctx: ForgeContext instance
            msl_source: Metal Shading Language source string
            fn_name: Entry point function name in the kernel
        """
        self._ptr = _lib.forge_kernel_compile(
            ctx.ptr,
            msl_source.encode("utf-8"),
            fn_name.encode("utf-8"),
        )
        if not self._ptr:
            _check_error(f"Failed to compile kernel '{fn_name}'")
            raise ForgeError(f"forge_kernel_compile returned NULL for '{fn_name}'")

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._ptr:
            _lib.forge_kernel_destroy(self._ptr)
            self._ptr = None

    @property
    def ptr(self):
        if not self._ptr:
            raise ForgeError("Kernel already destroyed")
        return self._ptr


class ForgePipeline:
    """Pipeline builder: queue dispatch ops then execute on GPU."""

    def __init__(self, ctx):
        self._ptr = _lib.forge_pipeline_create(ctx.ptr)
        if not self._ptr:
            _check_error("Failed to create pipeline")
            raise ForgeError("forge_pipeline_create returned NULL")
        self._executed = False

    def dispatch_1d(self, kernel, buffers, thread_count):
        """Queue a 1D kernel dispatch.

        Args:
            kernel: ForgeKernel instance
            buffers: list of ForgeBuffer instances (bound to [[buffer(0)]], [[buffer(1)]], ...)
            thread_count: total number of GPU threads
        """
        if self._executed:
            raise ForgeError("Pipeline already executed (pipelines are single-use)")
        n_bufs = len(buffers)
        buf_arr = (ctypes.c_void_p * n_bufs)(*[b.ptr for b in buffers])
        rc = _lib.forge_pipeline_dispatch_1d(
            self._ptr,
            kernel.ptr,
            buf_arr,
            ctypes.c_uint32(n_bufs),
            ctypes.c_uint64(thread_count),
        )
        if rc != 0:
            _check_error("dispatch_1d failed")

    def execute(self):
        """Execute pipeline on GPU. Blocks until complete. Returns gpu_time_ms."""
        if self._executed:
            raise ForgeError("Pipeline already executed")
        gpu_time = ctypes.c_double(0.0)
        rc = _lib.forge_pipeline_execute_timed(self._ptr, ctypes.byref(gpu_time))
        self._executed = True
        self._ptr = None  # Pipeline is consumed after execute
        if rc != 0:
            _check_error("Pipeline execution failed")
        return gpu_time.value

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._ptr and not self._executed:
            _lib.forge_pipeline_destroy(self._ptr)
            self._ptr = None
